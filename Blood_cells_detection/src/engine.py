import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calculate_map(model, dataloader, device, class_metrics=False):
    """Подсчет метрики mAP"""
    metric = MeanAveragePrecision(class_metrics=class_metrics)
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [{
                'boxes': out['boxes'].cpu(),
                'scores': out['scores'].cpu(),
                'labels': out['labels'].cpu()
            } for out in outputs]

            target = [{
                'boxes': t['boxes'].cpu(),
                'labels': t['labels'].cpu()
            } for t in targets]

            metric.update(preds, target)
            
    result = metric.compute()
    del metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_one_epoch(model, dataloader, optimizer, device):
    """Обучение модели в течение одной эпохи"""
    model.train()
    total_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, model_name="model", n_epochs=50):
    """Полный цикл обучения модели с сохранением лучших весов модели по метрике mAP_75"""
    best_map = 0.0
    
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Валидация
        metrics = calculate_map(model, val_loader, device)
        current_map_75 = metrics['map_75'].item()
        
        # Шаг планировщика на основе mAP_75
        scheduler.step(current_map_75)

        # Сохранение весов при улучшении метрики
        if current_map_75 > best_map:
            best_map = current_map_75
            save_path = f"models/best_{model_name}.pth"
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mAP: {metrics['map'].item():.4f} | "
            f"mAP_50: {metrics['map_50'].item():.4f} | "
            f"mAP_75: {current_map_75:.4f}\n"
        )

    # Очистка памяти после завершения всех эпох
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
