from ultralytics import YOLO

model = YOLO("Models/nano vehicle.pt")
print("model loaded")


def predict_and_annotate(im):
    results = model(im)

    for result in results:
        annotated_image = result.plot()

    return annotated_image


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    annotated_image = predict_and_annotate(image)
    cv2.imwrite(args.output, annotated_image)

    print("Done")
