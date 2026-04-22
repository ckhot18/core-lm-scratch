import math
import random

alphabet = " abcdefghijklmnopqrstuvwxyz"

# TOKENIZER: Maps a character to a unique integer ID (e.g., 'a' -> 1).
char_to_id = {c: i for i, c in enumerate(alphabet)}

# DE-TOKENIZER: Maps an ID back to a character (e.g., 1 -> 'a').
id_to_char = {i: c for i, c in enumerate(alphabet)}

# Weights, the model will adjust these numbers to minimize error.
weights = [[random.uniform(-0.1, 0.1) for _ in range(27)] for _ in range(27)]

def softmax(logits):
    """
    CONCEPT: Competitive Activation
    Takes raw scores (logits) and turns them into probabilities that sum to 1.0.
    """
    # THE MAX TRICK: We subtract the max value for numerical stability.
    # This prevents math.exp from exploding (vital for limited hardware).
    max_logit = max(logits)
    
    # Exponentiate every score: e^z
    exps = [math.exp(l - max_logit) for l in logits]
    
    # Sum all exponents for the denominator
    sum_exps = sum(exps)
    
    # Divide each by the sum to get the probability (0.0 to 1.0)
    return [e / sum_exps for e in exps]

def train(text, epochs=50, lr=0.1):
    """
    CONCEPT: The Training Loop (Optimization)
    The model guesses, calculates how wrong it was, and updates the weights.
    """
    for epoch in range(epochs):
        total_loss = 0
        samples_processed = 0
        
        # SLIDING WINDOW: Iterate through the text 2 characters at a time.
        for i in range(len(text) - 1):
            char1, char2 = text[i].lower(), text[i+1].lower()
            
            # Skip characters not in our alphabet
            if char1 not in char_to_id or char2 not in char_to_id:
                continue
            
            # Get the IDs (Indices)
            ix1, ix2 = char_to_id[char1], char_to_id[char2]
            
            # --- STEP 1: FORWARD PASS ---
            # In a Bigram, the "Input" is just the row of the current character.
            logits = weights[ix1]
            
            # Turn the row scores into probabilities
            probs = softmax(logits)
            
            #  STEP 2: LOSS CALCULATION 
            # We use Negative Log Likelihood (NLL).
            # If the model predicted 0.01 for the correct char, the loss is high.
            # If it predicted 0.99, the loss is near zero.
            correct_char_prob = probs[ix2]
            total_loss += -math.log(correct_char_prob)
            samples_processed += 1
            
            #  STEP 3: BACKWARD PASS (GRADIENT DESCENT) 
            # The derivative of (Softmax + Cross Entropy) is: (predictions - targets).
            # It's surprisingly simple math for such a powerful concept.
            for j in range(27):
                # Our target is 1.0 for the correct next character, 0.0 for others.
                target = 1.0 if j == ix2 else 0.0
                
                # How far off was the prediction?
                gradient = probs[j] - target
                
                #  STEP 4: UPDATE 
                # Move the weight in the opposite direction of the error.
                # 'lr' is the Learning Rate (the step size).
                weights[ix1][j] -= lr * gradient
                
        # Print progress every 10 iterations
        if epoch % 10 == 0:
            avg_loss = total_loss / samples_processed
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

def generate(length=100):
    """
    CONCEPT: Inference
    The model "walks" through the weight matrix to create a sentence.
    """
    current_ix = 0 # Start with a space
    output = ""
    
    for _ in range(length):
        # Get probabilities for the current character
        probs = softmax(weights[current_ix])
        
        # SAMPLING: Roll a weighted die based on the probabilities.
        # This is what makes AI "creative" rather than just a fixed script.
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                current_ix = i
                break
        
        output += id_to_char[current_ix]
    return output

#  RUNNING THE DRAFT 
training_data = '''
the sun rises in the east and sets in the west the sky turns orange and red as the day ends the wind moves softly across the land and the trees sway with it the birds fly across the sky and sing their simple songs the world moves in patterns that repeat again and again
this is a simple dataset for a small language model it is not complex but it is enough to learn patterns in text the goal is to help the model understand how characters follow each other and how words are formed in a basic way
the cat sits on the mat and looks at the door the dog runs in the yard and barks at the sky the man walks down the road and thinks about his day the child plays in the field and laughs with joy
learning happens step by step the model reads the text and counts each pair of characters it sees over time it builds a map of what comes next after each letter this map helps it generate new sequences that feel similar to the training data
the more data you give the model the better it becomes but even a small dataset like this can show how the system works and how patterns emerge from rpetition and structure
the sun rises in the east and sets in the west the sky turns orange and red as the day ends the wind moves softly across the land and the trees sway with it the birds fly across the sky and sing their simple songs
this is only the beginning of building a language model there are many improvements that can be made such as using more context better tokenization and more advanced probability methods but this step is important to understand the foundation
the cat sits on the mat the dog runs in the yard the man walks down the road the child plays in the field these simple sentences help reinforce common patterns in the data and make the model more stable
repeat the patterns again and again so the model learns clearly the more consistent the data the easier it is for the model to predict what comes next and generate readable output
the sun rises in the east and sets in the west the sky turns orange and red as the day ends the wind moves softly across the land and the trees sway with it the birds fly across the sky and sing their simple songs
'''
train(training_data, epochs=100, lr=0.5)
print("\n[AI GENERATION]:")
print(generate(300))