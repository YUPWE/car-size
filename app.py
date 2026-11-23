import gradio as gr

from src.system import CarSizer

sizer = CarSizer()
cfg = sizer.cfg

MESSAGES = {
    "big": "**BIG** (SUV / pickup / van): send to the large-bay level.",
    "small": "**SMALL** (sedan / hatchback / coupe / convertible / wagon): compact bays are fine.",
    "not a car": "This doesn't look like a car.",
}


def classify(image):
    if image is None:
        return "Upload a photo first.", {}
    out = sizer.predict(image)
    text = MESSAGES[out["decision"]]
    text += f"\n\np(big) = {out['p_big']:.2f} (threshold {cfg['big_threshold']}) · CLIP p(not a car) = {out['p_not_car']:.2f}"
    return text, {"big": out["p_big"], "small": 1 - out["p_big"]}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Photo of one car"),
    outputs=[gr.Markdown(label="Decision"), gr.Label(label="Probability")],
    title="Big car or small car? ResNet-18 + CLIP",
    description=("Classifies a car as BIG (SUV, pickup, van, minivan) or SMALL (sedan, hatchback, coupe, "
                 "convertible, wagon). The big threshold is set low on purpose so big cars are rarely missed. "
                 "CLIP rejects photos that are not cars."),
)

if __name__ == "__main__":
    demo.launch()
