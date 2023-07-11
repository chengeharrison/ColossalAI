import argparse
import uuid
import gradio as gr
from enum import auto, Enum
import dataclasses
from typing import List
import random
import time
import torch
import transformers
from gradio_css import code_highlight_css
from chat import load_model


class SeparatorStyle(Enum):
    ADD_BOS_EOS_TOKEN = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    model: transformers.AutoModelForCausalLM
    tokenizer: transformers.AutoTokenizer
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.ADD_BOS_EOS_TOKEN
    sep: str = "</s>"

    skip_next: bool = False
    
    def clear(self):
        self.messages=[]

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
    
    def save_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>\n"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            model=self.model,
            tokenizer=self.tokenizer,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep
        }
        
block_css = (
        code_highlight_css
        + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}
"""
)

model, tokenizer = load_model("/data3/data/SFT_improvement/Coati-SFT-v2-433k-epoch2-3-steps-6.77k/")
# model = None
# tokenizer = None
conv = Conversation(
    model=model,
    tokenizer=tokenizer,
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    sep="</s>",
)

def get_default_conv_template(model_name=None):
    if model_name is None:
        return default_conversation
    model_name = model_name.lower()
    if "phoenix" in model_name or "chimera" in model_name:
        return default_conversation
    else:
        raise NotImplementedError

default_conversation = conv

def set_global_vars(state_, model_, tokenizer_):
    global state, model, tokenizer
    state = state_
    model = model_
    tokenizer = tokenizer_

def add_text(state, user_message, history):
    # print(111)
    if state is None:
        state = get_default_conv_template().copy()
        
    if len(user_message) <= 0:
        state.skip_next = True
        return state, "", state.to_gradio_chatbot()
    
    state.append_message(state.roles[0], user_message)
    state.append_message(state.roles[1], None)
    return state, "", state.to_gradio_chatbot()

# def bot(history):
#     bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#     history[-1][1] = ""
#     for character in bot_message:
#         history[-1][1] += character
#         time.sleep(0.05)
#         yield history

@torch.inference_mode()
def bot(
    state, chatbot, temperature, max_new_tokens
):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield state, state.to_gradio_chatbot()
        return

    # yield state, state.to_gradio_chatbot()
    # return
    
    model = state.model
    tokenizer = state.tokenizer
    # yield state, state.to_gradio_chatbot()    
    if len(state.messages) == state.offset + 2:
        new_state = conv
        new_state._id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
        
    # Construct prompt
    prompt = state.get_prompt()
    state.messages[-1][-1] = "▌"
    
    yield state, state.to_gradio_chatbot()
        
    device = torch.cuda.current_device()
    context_len = 2048
    stream_interval = 2
    
    stop_str = "</s>"
    stop_token_ids = [tokenizer.eos_token_id]

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)
    
    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 15

    input_ids = input_ids[-max_src_len:]
    
    # output = model.generate(torch.as_tensor([input_ids], device=device), **generation_kwargs)
    
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        
        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                
                state.messages[-1][-1] = output + "▌"
                
                yield state, state.to_gradio_chatbot()
            else:
                raise NotImplementedError

        if stopped:
            break
        
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield state, state.to_gradio_chatbot()
    
    del past_key_values
    
def clear_history(state):
    state.messages = []
    state = None
    return state, [], ""

def load_demo_single():

    state = None
    return (
        state,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Button.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo():
    return load_demo_single()

def build_web_ui():
    notice_markdown = ("""ColossalChat Demo
                       
                       具备多轮对话能力""")

    learn_more_markdown = ("""
    Find us 
    """)
    # model, tokenizer = load_model(args.model_path)
    state = gr.State()
    # x = gr.State({"model":model, "tokenizer":tokenizer})
    # x = {"model":model, "tokenizer":tokenizer}
    # model._id = 0
    # tokenizer._id = 1
    model = gr.State()
    tokenizer = gr.State()

    
    # with gr.Blocks() as demo:
    gr.HTML("""
            <center>
            <img src="file/logo_coati1.png"/>
            </center>
            """)
    
    # with gr.Row():
    #     img = gr.Image("https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/logo_coati.png")
    # notice = gr.Markdown(notice_markdown)
        
    chatbot = gr.Chatbot(height=600,show_label=False)
    with gr.Row():
        with gr.Column(scale=18):
            msg = gr.Textbox(
                show_label=False,
                placeholder="开始聊天吧",
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="发送")
        with gr.Column(scale=2, min_width=50):
            clear_btn = gr.Button(value="🗑️ 清除历史")
        
    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.0,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        max_new_tokens = gr.Slider(
            minimum=0,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max new tokens",
    )
            
    msg.submit(add_text, [state, msg, chatbot], [state, msg, chatbot], queue=False).then(
        bot,
        [state, chatbot, temperature, max_new_tokens],
        [state, chatbot],
    )
        
    send_btn.click(add_text, [state, msg, chatbot], [state, msg, chatbot], queue=False).then(
        bot,
        [state, chatbot, temperature, max_new_tokens],
        [state, chatbot],
    )
    
    clear_btn.click(clear_history, [state], [state, chatbot, msg], queue=False)
        
    # response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    # gr.Markdown(learn_more_markdown)
    # demo.queue()
    # demo.launch(share=True)
    
    return state, chatbot, msg, send_btn, clear_btn, parameter_row

def build_demo(args):
    with gr.Blocks(
            title="ColossalChat",
            theme=gr.themes.Base(),
            css=block_css,
    ) as demo:
        (
            state,
            chatbot,
            textbox,
            send_btn,
            clear_btn, 
            parameter_row,
        ) = build_web_ui()

        demo.load(
                load_demo,
                [],
                [
                    state,
                    chatbot,
                    textbox,
                    send_btn,
                    clear_btn, 
                    parameter_row,
                ]
            )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/data3/data/SFT_improvement/Coati-SFT-v1-282k-epoch3-3-steps-6.618k/")
    args = parser.parse_args()
    
    # state = conv.copy()
    
    # model, tokenizer = load_model(
    #     args.model_path
    # )
    # set_global_vars(state, model, tokenizer)
    demo = build_demo(args)
    demo.queue()
    demo.launch(share=True)