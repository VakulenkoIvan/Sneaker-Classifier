import json
import torch
from PIL import Image
from torchvision import transforms
from model import get_model


class SneakerClassifier:
    def __init__(self, weights_path: str = "best_model.pth", classes_path: str = "classes.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # загружаем названия всех 50 классов
        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        # создаём архитектуру и загружаем обученные веса
        self.model = get_model(num_classes=len(self.classes))
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # обязательно выключаем Dropout и фиксируем BatchNorm

        # те же строгие трансформации, что были на валидации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_source, top_k: int = 3) -> list[tuple[str, float]]:
        # открываем картинку (работает и с путём к файлу, и с io.BytesIO от Telegram)
        image = Image.open(image_source).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # [3, 224, 224] -> [1, 3, 224, 224]

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            # делаем название красивым: "nike_air_force_1_low" -> "Nike Air Force 1 Low"
            raw_name = self.classes[idx.item()]
            clean_name = raw_name.replace("_", " ").title()
            results.append((clean_name, round(prob.item() * 100, 1)))

        return results


if __name__ == "__main__":
    # быстрый локальный тест на любой картинке из датасета
    classifier = SneakerClassifier()
    test_img = "sneakers-dataset/nike_air_force_1_low/0001.jpg"
    predictions = classifier.predict(test_img)

    print("Результат распознавания:")
    for rank, (name, conf) in enumerate(predictions, 1):
        print(f"{rank}. {name} — {conf}%")