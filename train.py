from utils.dailydialog_loader import load_daily_dialog
from src.semantic_captioning_character_lstm_decoder import SemanticCaptioningCharacterLSTMDecoder

contexts,questions = zip(*load_daily_dialog(path_to_file="data/daily_dialog",include_full_context=False))
decoder = SemanticCaptioningCharacterLSTMDecoder(recursion_depth=100,path_to_model_weights="pretrained_models/daily_dialog.hdf5")
decoder.train(
    question_contexts=contexts,
    questions=questions,
    batch_size=10,
    epochs=100,
    save_to_file_path="pretrained_models/daily_dialog.hdf5"
) 