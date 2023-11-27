def predict_and_annotate(model, im):
    results = model(im)

    for result in results:
        annotated_image = result.plot()

    return annotated_image


def convert_to_pil_image(cv2_image):
    from PIL import Image
    import cv2
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    from ultralytics import YOLO
    import argparse
    import cv2

    model = YOLO("Models/nano vehicle.pt")
    print("model loaded")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    annotated_image = predict_and_annotate(model, image)
    cv2.imwrite(args.output, annotated_image)

    print("Done")
