from models import Naive_LLM



if __name__ == "__main__":
    test_model = Naive_LLM("openlm-research/open_llama_3b_v2")
    
    # Forward is for training
    # You can give a single string
    loss = test_model.forward("Cheer up guys")
    print(loss) # 5.3284
    # Or You can give a list of string
    loss = test_model.forward(["HPML 6998 is Fun", "Cheer up guys", "Test example for Forward"])
    print(loss) # 6.3743
    
    # You can also generate sequence by giving input_sequence
    
    # You can give a single string (with additional generate kwargs)
    generated, tps, elapsed = test_model.generate("Please introduce yourself: ")
    print(f"Generated: {generated}, Tokens/sec: {tps} ({elapsed} seconds)") # The return value will be a string and include the prompt also
    # Or You can give a list of string (with additional generate kwargs)
    generated, tps, elapsed = test_model.generate(["This is for test generate any word", "Test is always", "Hi all, I am"])
    print(f"Generated: {generated}, Tokens/sec: {tps} ({elapsed} seconds)") # The return value will be a list and include the prompt also
    
    # If you want to exclude the prompt from the output just give return_full_text as False
    generated, tps, elapsed = test_model.generate("Please introduce yourself: ", return_full_text=False)
    print(f"Generated: {generated}, Tokens/sec: {tps} ({elapsed} seconds)") # The return value will be a string and exclude the prompt
    generated, tps, elapsed = test_model.generate(["This is for test generate any word", "Test is always", "Hi all, I am"], return_full_text=False) # The return value will be a list and exclude the prompt
    print(f"Generated: {generated}, Tokens/sec: {tps} ({elapsed} seconds)")