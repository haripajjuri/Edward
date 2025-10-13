import sentencepiece as spm
spm.SentencePieceTrainer.train(
    f"--input=tokenizer_text.txt "
    f"--model_prefix=tokenizer "
    f"--vocab_size=12000 "
    f"--model_type=bpe "
    f"--pad_id=0 --pad_piece=<pad> "
    f"--bos_id=1 --bos_piece=<bos> "
    f"--eos_id=2 --eos_piece=<eos> "
    f"--unk_id=3 --unk_piece=<unk> "
    f"--user_defined_symbols="
    "<human>,<robot>,<nl>,!,?,@,#,$,%,^,&,*,(,),_,+,-,=,{,},[,],|,\\,:,;,\",',<,>,.,/,≠,≥,≤,%,¥,₹,€,£"
)
#
#from transformers import LlamaTokenizer
#tokenizer = LlamaTokenizer(
#    vocab_file="tokenizer.model",
#    eos_token="<eos>",
#    bos_token="<bos>",
#    pad_token="<pad>",
#    unk_token="<unk>",   
#)
#tokenizer.add_special_tokens({"additional_special_tokens": ["<human>", "<robot>"]})
#tokenizer.add_eos_token=True
#tokenizer.save_pretrained("./my_tokenizer")