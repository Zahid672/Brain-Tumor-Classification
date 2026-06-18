import os 
import glob 
import shutil
import random

def split_data_into_test_train(main_dir, split_ratio=0.8):
    """
    Split the data into training and testing datasets.
    Args:
        main_dir: Path to main directory containing class folders
        split_ratio: Ratio for train/test split (default 0.8 = 80% training)
    """
    # Create train and test directories
    train_dir = os.path.join(main_dir, 'train')
    test_dir = os.path.join(main_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of class directories
    class_dirs = [d for d in os.listdir(main_dir) 
                 if os.path.isdir(os.path.join(main_dir, d))
                 and d not in ['train', 'test']]

    for class_name in class_dirs:
        # Create class subdirectories in train and test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all files for this class
        class_path = os.path.join(main_dir, class_name)
        files = glob.glob(os.path.join(class_path, '*'))
        
        # Shuffle files randomly
        random.shuffle(files)
        
        # Calculate split point
        split_point = int(len(files) * split_ratio)
        
        # Split into train and test sets
        train_files = files[:split_point]
        test_files = files[split_point:]

        # Copy files to respective directories
        for src in train_files:
            dst = os.path.join(train_class_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            
        for src in test_files:
            dst = os.path.join(test_class_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    print(f"Data split complete: {len(class_dirs)} classes processed")
    print(f"Split ratio: {split_ratio:.0%} training, {(1-split_ratio):.0%} testing")



if __name__ == "__main__":
    main_dir = os.path.join('BrainTumorDatasets', 'BT-small-2c-dataset-3000im')
    split_data_into_test_train(main_dir, split_ratio=0.8)