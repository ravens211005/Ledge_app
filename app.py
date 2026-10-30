import config  # first: hides GPUs before torch loads

import logging
import time

import gradio as gr
import numpy as np

import hazard_analysis
import inference
import visualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

DISCLAIMER = ("**Research prototype.** Not a validated safety-critical system. Depth values are "
              "*relative* (larger = closer, normalised per image) and are not distances in metres.")
EMPTY = (None, None, None, None, None, None, [], "")


def analyse(pil_img, conf):
    t_start = time.perf_counter()
    try:
        rgb = inference.prepare_image(pil_img)
        res = inference.load_models().run(rgb, conf=conf)

        rows = hazard_analysis.region_stats(res.detections, res.depth_rel)

        frame_bgr = np.ascontiguousarray(rgb[..., ::-1])
        floor = hazard_analysis.floor_mask_from_detections(
            res.detections, hazard_analysis.FLOOR_CLASS_NAMES, rgb.shape)
        hz = hazard_analysis.detect_fall_hazard(frame_bgr, floor, res.depth_raw)
        hazard_img = np.ascontiguousarray(hz.overlay_bgr[..., ::-1])

        total = time.perf_counter() - t_start
        t = res.timings
        lines = [f"Processed in **{total:.2f} s** (YOLO {t['yolo_s']:.2f} s, depth {t['depth_s']:.2f} s, "
                 f"hazard {hz.timing_ms / 1000:.2f} s), working size {rgb.shape[1]}x{rgb.shape[0]}."]
        n_st = sum(d.class_name == config.STAIR_CLASS_NAME for d in res.detections)
        n_ed = sum(d.class_name == config.EDGE_CLASS_NAME for d in res.detections)
        lines.append(f"Detections: {n_st} stairs, {n_ed} edges (conf >= {float(conf):.2f}).")
        if not res.detections:
            lines.append("⚠️ No staircase or edge detected. Uncertain — unable to assess reliably.")
        log.info("analysed %s conf=%.2f total=%.2fs hazard=%s %s", rgb.shape, float(conf), total, hz.status, t)

        return (rgb, visualize.segmentation_overlay(rgb, res.detections),
                visualize.edge_view(rgb, res.detections), visualize.depth_colormap(res.depth_rel),
                visualize.combined(rgb, res.detections, res.depth_rel), hazard_img, rows,
                hz.summary_markdown() + "\n\n" + "\n\n".join(lines))
    except inference.InputError as e:
        return (*EMPTY[:-1], f"⚠️ {e}")
    except Exception as e:  # never crash the UI
        log.exception("analysis failed")
        return (*EMPTY[:-1], f"❌ Inference error: {type(e).__name__}: {e}")


def build_ui():
    with gr.Blocks(title="Staircase Analysis (research prototype)") as demo:
        gr.Markdown("# Staircase Navigation & Hazard Analysis")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload staircase image (JPG / PNG)", sources=["upload"])
            with gr.Column():
                conf = gr.Slider(0.05, 0.90, value=config.YOLO_CONF, step=0.05,
                                 label="YOLO confidence threshold")
                run_btn = gr.Button("Analyse Image", variant="primary")
                clear_btn = gr.ClearButton(value="Clear")
        with gr.Row():
            o_orig = gr.Image(label="Original", interactive=False)
            o_seg = gr.Image(label="YOLO segmentation", interactive=False)
            o_edge = gr.Image(label="Stair edges", interactive=False)
        with gr.Row():
            o_depth = gr.Image(label="Depth map (relative; brighter = closer)", interactive=False)
            o_comb = gr.Image(label="Combined annotated output", interactive=False)
            o_haz = gr.Image(label="Fall-hazard overlay", interactive=False)
        report = gr.Markdown()
        table = gr.Dataframe(headers=hazard_analysis.TABLE_HEADERS, label="Detected regions "
                             "(relative depth, not metres)", interactive=False)

        run_btn.click(analyse, inputs=[inp, conf],
                      outputs=[o_orig, o_seg, o_edge, o_depth, o_comb, o_haz, table, report])
        clear_btn.add([inp, o_orig, o_seg, o_edge, o_depth, o_comb, o_haz, table, report])
    return demo


if __name__ == "__main__":
    inference.load_models()  # load once at startup (includes warm-up)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=config.QUEUE_MAX_SIZE)
    demo.launch(server_name=config.HOST, server_port=config.PORT, show_error=True)