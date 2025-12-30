import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. DATA GENERATION (Synthetic Email Data - 1000 Samples)
# ==============================================================================
# Scenario: 1000 Emails. 50 different words (Features).
np.random.seed(42)
n_samples = 1000
n_features = 50 

# Generate random word counts (between 0 and 4)
X = np.random.randint(0, 5, size=(n_samples, n_features))

# Generate random labels (0: Normal, 1: Spam)
y = np.random.randint(0, 2, size=n_samples)

# --- SIGNAL INJECTION (To help models learn) ---
# For Spam (1) messages, we increase the count of the first 10 words.
# This creates a strong pattern: "High frequency of these words = SPAM"
X[y == 1, :10] += 3 

# ==============================================================================
# 2. TRAIN - TEST SPLIT
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==============================================================================
# 3. HELPER FUNCTION (REPORT GENERATOR) 🖨️
# ==============================================================================
def print_report(model_name, train_acc, test_acc):
    print(f"\n📢 {model_name.upper()} REPORT CARD:")
    print("-" * 50)
    print(f"   📘 Training Score: {train_acc * 100:.2f}%")
    print(f"   📝 Test Score:     {test_acc * 100:.2f}%")
    
    diff = train_acc - test_acc
    print("   🔍 DIAGNOSIS: ", end="")
    
    if diff > 0.10:
        print("⚠️ OVERFITTING DETECTED!")
        print("      The model memorized the training data but failed on the test data.")
    elif train_acc < 0.60:
        print("⚠️ UNDERFITTING DETECTED!")
        print("      The model failed to learn the patterns (Scores are too low).")
    elif test_acc > train_acc:
        print("✅ EXCELLENT (ROBUST MODEL)")
        print("      Performs even better on unseen data (Lucky distribution).")
    else:
        print("✅ HEALTHY & BALANCED")
        print("      Train and Test scores are close. The model generalizes well.")
    print("=" * 50)

# ==============================================================================
# 4. MODEL 1: MULTINOMIAL NAIVE BAYES (Frequency Based) 🔢
# ==============================================================================
# Focuses on HOW MANY times a word appears (Frequency).
# It loves the "+= 3" signal we added earlier.
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

mnb_train = mnb.score(X_train, y_train)
mnb_test = mnb.score(X_test, y_test)

print_report("Multinomial NB (Counter)", mnb_train, mnb_test)

# ==============================================================================
# 5. MODEL 2: BERNOULLI NAIVE BAYES (Binary / Existence Based) 💡
# ==============================================================================
# Ignores counts. Checks: Is count > 0? (Yes/No).
# It treats '3' and '1' exactly the same.
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

bnb_train = bnb.score(X_train, y_train)
bnb_test = bnb.score(X_test, y_test)

print_report("Bernoulli NB (Binary)", bnb_train, bnb_test)