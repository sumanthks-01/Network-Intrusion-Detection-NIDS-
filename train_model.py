from model_trainer import IDSModelTrainer

if __name__ == "__main__":
    print("Training Network Intrusion Detection Model...")
    
    trainer = IDSModelTrainer()
    trainer.train_model('data/combined_cleaned_dataset.csv', test_size=0.2)
    trainer.save_model('ids_model.pkl')
    
    print("\nModel training completed!")
    print("Attack types the model can detect:")
    for i, attack_type in enumerate(trainer.label_encoder.classes_):
        clean_name = str(attack_type).encode('ascii', 'ignore').decode('ascii')
        print(f"{i+1}. {clean_name}")