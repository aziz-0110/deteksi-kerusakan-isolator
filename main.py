import os
import cv2
from ultralytics import YOLO
import gc  # Garbage collector
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

class Controller:
    def __init__(self):
        model_path_yolo = 'datasets/dataset-kaggle/split/runs/weights/best.pt'
        self.model_yolo = YOLO(model_path_yolo)

        model_path_cnn = 'model_isolator.pth'

        # Kelas sesuai urutan folder dataset kamu
        self.classes = ['Pollution', 'Cracked']

        # Load model CNN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_cnn = models.resnet18(pretrained=False)
        self.model_cnn.fc = nn.Linear(self.model_cnn.fc.in_features, 2)  # 2 kelas: kotor dan pecah
        self.model_cnn.load_state_dict(torch.load(model_path_cnn, map_location=self.device))
        self.model_cnn = self.model_cnn.to(self.device)
        self.model_cnn.eval()

        # Preprocess gambar
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        path_img_predic = 'datasets/images/'
        output_dir = 'hasil'
        self.predic_yolo(path_img_predic, output_dir)



    def predic_yolo(self, path, output_dir):
        batch_size = 20  # Sesuaikan dengan kapasitas RAM
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i in range(0, len(image_files), batch_size):
        # for i in range(0, len(image_files[:2])):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [os.path.join(path, f) for f in batch_files]

            results = self.model_yolo.predict(batch_paths, conf=0.5)
            # results = self.model_yolo.predict(batch_paths[:1], conf=0.5)

            for j, r in enumerate(results):
                img_path = batch_paths[j]
                img_ori = cv2.imread(img_path)
                if img_ori is None:
                    continue

                base_name = os.path.splitext(batch_files[j])[0]

                for k, (box, cls) in enumerate(zip(r.boxes.xyxy, r.boxes.cls)):
                    if int(cls) == 1:
                        x1, y1, x2, y2 = map(int, box)
                        cropped = img_ori[y1:y2, x1:x2]
                        if cropped.size > 0:
                            label = self.predic_cnn(cropped)
                            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            cv2.putText(img_ori, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                cv2.imwrite(f"{output_dir}/{base_name}.png", img_ori)
                # img_ori = cv2.resize(img_ori, (1080, 720))
                # cv2.imshow('Hasil Prediksi', img_ori)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Cleanup
                del img_ori

            del results
            gc.collect()  # Bersihkan memori setiap batch

        del self.model_yolo
        gc.collect()
        print('Selesai')

    def predic_cnn(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img_rgb)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model_cnn(input_tensor)
            _, predicted = torch.max(output, 1)
            label = self.classes[predicted.item()]

        print(f"Prediksi: {label}")
        return label

if __name__ == '__main__':
    Controller()
