import torch

def main():
    print("=" * 40)
    print("Проверка готовности видеокарты (GPU)")
    print("=" * 40)
    
    print(f"Версия PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\n✅ УРА! Ваша видеокарта готова к работе!")
        print(f"✅ Устройство: {torch.cuda.get_device_name(0)}")
        print(f"✅ Версия CUDA: {torch.version.cuda}")
        print("\nТеперь все скрипты (extract_faces.py, cache_features.py и т.д.)")
        print("автоматически подхватят вашу RTX 4050 и будут летать!")
    else:
        print("\n❌ Увы, PyTorch все еще не видит видеокарту.")
        print("Пожалуйста, дождитесь окончания фоновой установки библиотеки в терминале.")
        print("После того как пакет скачается и установится, запустите этот скрипт еще раз.")

if __name__ == "__main__":
    main()
