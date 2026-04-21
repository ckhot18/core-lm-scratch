// What to create ? 
// A table (matrix) where rows represent current character and col represent next character
// Given character A, what is the most likely character B?

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <random>

class BigramModel{
    private:
        std::string alphabet = " abcdefghijklmnopqrstuvwxyz"; // vocaboulary , possible tokens (26 letters + space)
        std::map<char, int> char_to_id; // tokenizer : this map translates map to indexes, eg a to 0 b to 1 and so on, cause machine doesnt understand letters but numbers
        std::vector<std::vector<float>> counts;// knowledge matrix, rows - current char, col - next char prediction
    public:
        //constructor
        BigramModel() : counts(27, std::vector<float>(27, 0.0f)){
            // to populate the map, (preprocessing)
            for(int i = 0; i < alphabet.size(); i++){
                char_to_id[alphabet[i]] = i;
            }
        }

         //training loop
         void train(const std::string& filepath){
            std::ifstream file(filepath);
            if(!file){
                std::cerr << "Error : File not found";
                return;
            }

            std::string line;
            while(std::getline(file, line)){
                for(size_t i = 0; i<line.length() - 1; i++){
                    // we force everything to lowercase // this is called data normalization
                    char c1 = tolower(line[i]);
                    char c2 = tolower(line[i+1]);
                    // i and i+1 logic to create a sliding window

                    if(char_to_id.count(c1) && char_to_id.count(c2)){
                        int row = char_to_id[c1];
                        int col = char_to_id[c2];
                        counts[row][col] += 1.0f;
                        /*
                            This is the actual learning.
                            Every time the model sees 'h' followed by 'e' in your text file, it goes to the 'h' row, the 'e' column,
                            and adds a "point." 
                            After reading a whole book, the 'h' row will have a massive number in the 'e' column 
                            and almost nothing in the 'z' column.
                        */

                    }
                }
            }
            std::cout << "[SUCCESS] Knowledge matrix built from training data.\n";

         }

         void generate(int length) {
    // 1. Initialize the Random Number Engine
    // mt19937 is a high-quality Mersenne Twister generator. 
    // random_device{}() provides a "seed" so the output is different every run.
    std::mt19937 gen(std::random_device{}());

    // 2. State Management
    // We start with index 0 (the space). In Bigram logic, this asks: 
    // "What letter usually starts a word?"
    int current_idx = 0; 

    std::cout << "Generated Text: ";

    // 3. The Generation Loop
    // We run this 'length' times to generate that many characters.
    for (int i = 0; i < length; ++i) {
        
        /* 4. THE CORE AI LOGIC: Weighted Sampling
         * We look at the row in our 'counts' matrix for the current character.
         * counts[current_idx] is a std::vector<float> containing 27 numbers.
         * discrete_distribution turns these raw counts into a weighted lottery.
         */
        std::discrete_distribution<> dist(counts[current_idx].begin(), counts[current_idx].end());

        // 5. The "Dice Roll"
        // We pass our random generator 'gen' to the distribution.
        // It returns an integer (0-26) based on the weights in that row.
        int next_idx = dist(gen);

        // 6. De-tokenization
        // Convert the number back to a character and print it.
        std::cout << alphabet[next_idx];

        // 7. Update State
        // This is a "Markov Chain." The character we just picked becomes 
        // the "current" character for the next iteration.
        current_idx = next_idx;
    }
    std::cout << std::endl;
}

};

int main() {
    BigramModel model;
    
    model.train("text.txt"); 
    model.generate(200);
    return 0;
}