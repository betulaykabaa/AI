import matplotlib.pyplot as plt

# 1. Setup Priors (General email traffic)
p_spam = 0.40
p_normal = 0.60

# 2. Likelihoods (Probability of seeing "WIN" word)
p_word_given_spam = 0.80   # 80% of spam emails contain "WIN"
p_word_given_normal = 0.10 # Only 10% of normal emails contain "WIN"

# 3. Evidence: We see the word "WIN". What is the total chance of seeing it?
p_word = (p_spam * p_word_given_spam) + (p_normal * p_word_given_normal)

# 4. Bayes Update: Is it Spam given the word "WIN"?
p_spam_given_word = (p_word_given_spam * p_spam) / p_word

# --- Visualization ---
categories = ['Spam Probability\n(No Word Seen)', 'Spam Probability\n(Saw word "WIN")']
probs = [p_spam, p_spam_given_word]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, probs, color=['blue', 'orange'])

plt.title('Spam Filter: Updating Belief based on a Word')
plt.ylabel('Probability')
plt.ylim(0, 1)

# Add text labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1) # Decision boundary
plt.text(0.5, 0.52, 'Decision Threshold (0.5)', color='gray')

plt.show()