from training.trainer import load_tokenizer, load_training_dataset

def test_tokenizer():
    """Make sure we can encode and decode with the tokenizer"""
    tokenizer = load_tokenizer()
    assert tokenizer.decode(tokenizer.encode("Hello Dolly!")) == "Hello Dolly!"

def test_load_training_dataset():
    """Make sure we can load the training dataset and it has records"""
    dataset = load_training_dataset()
    assert dataset.num_rows > 50_000
