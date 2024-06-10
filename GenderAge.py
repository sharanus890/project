import shutil
import cv2
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoImageProcessor, AutoModelForImageClassification
import argparse
import os
from tqdm import tqdm

def process_frame(frame, face_cascade, age_model, age_transforms, gender_processor, gender_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):

        max_y_red = 80
        max_y_inc = 30

        while True:
            if y-max_y_red < 0:
                max_y_red = max_y_red - 10
            else:
                break

        while True:
            if y+max_y_inc > rgb_frame.shape[1]:
                max_y_inc = max_y_inc - 10
            else:
                break


        max_x_red = 30
        max_x_inc = 30

        while True:
            if x-max_x_red < 0:
                max_x_red = max_x_red - 10
            else:
                break

        while True:
            if x+max_x_inc > rgb_frame.shape[0]:
                max_x_inc = max_x_inc - 10
            else:
                break

        cropped_face = rgb_frame[y-max_y_red:y+h+max_y_inc, x-max_x_red:x+w+max_x_inc]

        # cropped_face = rgb_frame[y:y+h, x:x+w]

        im_pil = Image.fromarray(cropped_face.astype('uint8'), 'RGB')

        inputs = age_transforms(im_pil, return_tensors='pt')
        output = age_model(**inputs)

        proba = output.logits.softmax(1)
        preds = proba.argmax(1)
        predicted_age_label = age_model.config.id2label[preds.item()]

        inputs = gender_processor(cropped_face, return_tensors="pt")

        with torch.no_grad():
            logits = gender_model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        predicted_gender_label = gender_model.config.id2label[predicted_label]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{predicted_gender_label} - {predicted_age_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    age_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    age_transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
    gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    parser = argparse.ArgumentParser(description='Process images for gender and age detection')
    parser.add_argument('--inputdir', type=str, help='Path to the directory containing images')
    parser.add_argument('--savedir', type=str, help='Path to the directory to save the predicted images')
    parser.add_argument('--vidfile', type=str, help='Path to the video file')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam number') # Use 0 for the default webcam
    parser.add_argument('--vidsavepath', type=str, help='Path to save the predicted video')
    parser.add_argument('--vidname', type=str, help='Name of the predicted video')
    args = parser.parse_args()

    if args.inputdir:
        images_list = os.listdir(args.inputdir)
        if args.savedir:
            if os.path.exists(args.savedir):
                shutil.rmtree(args.savedir)
                os.makedirs(args.savedir)
            else:
                os.makedirs(args.savedir)
        for img_name in images_list:
            img_full_path = os.path.join(args.inputdir, img_name)
            frame = cv2.imread(img_full_path)
            processed_frame = process_frame(frame, face_cascade, age_model, age_transforms, gender_processor, gender_model)
            
            if args.savedir:
                save_path = os.path.join(args.savedir, img_name)
                cv2.imwrite(save_path, processed_frame)
            else:
                cv2.imshow('Processed Frame', processed_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print(f"Image {img_name} processed successfully.")

  
    elif args.vidfile:
        cap = cv2.VideoCapture(args.vidfile)
        sec = 20
        fw = int(cap.get(3))
        fh = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_in_new = total_frames // (fps * sec)
        progress_bar = tqdm(total=total_frames_in_new, desc='Processing Frames', unit='frame')
        
        if args.vidsavepath:
            if not os.path.exists(args.vidsavepath):
                os.makedirs(args.vidsavepath)
            if not args.vidname:
                args.vidname = 'predicted_video.mp4'
            save_path = os.path.join(args.vidsavepath, args.vidname)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps//sec if fps > 0 else 1, (fw,fh))
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % int(fps * sec) == 0:
                processed_frame = process_frame(frame, face_cascade, age_model, age_transforms, gender_processor, gender_model)
                cv2.imshow('Frame', processed_frame)
                if args.vidsavepath:
                    out.write(processed_frame)
                progress_bar.update(1)  # Update the progress bar
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        progress_bar.close()  # Close the progress bar
        cap.release()
        if args.vidsavepath:
            out.release()
        cv2.destroyAllWindows()


    else:
        cap = cv2.VideoCapture(args.webcam)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, face_cascade, age_model, age_transforms, gender_processor, gender_model)
            cv2.imshow('Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

