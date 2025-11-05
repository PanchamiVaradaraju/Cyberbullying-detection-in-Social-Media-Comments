# Cyberbullying-detection-in-Social-Media-Comments
A machine learning project to detect cyberbullying in social media comments

1. Abstract

Cyberbullying Detection in Social Media Comments

Problem Statement 

Cyberbullying has slowly turned into one of the biggest problems of our generation. Every day, people are getting hurt by words shared online, and the emotional damage keeps growing as social media becomes a bigger part of our lives. What makes it worse is that the systems made to detect this kind of abuse still don’t really understand how humans communicate. They look for harsh words or harsh language, but most of the time, the real pain comes from messages that sound normal but carry a hidden meaning. Because of this, two frustrating things happen all the time normal posts get flagged for no reason, while truly hurtful comments pass through without any warning. It shows that these systems are not thinking the way people do. What we really need is a detection method that can feel and interpret context something more human-aware.

Proposed Solution

My idea is to create a smarter machine learning model that can sense the tone, mood, and intent behind a message, not just read the words on the screen. I’ll be using a strong pre trained language model like NLP which is Natural Language Processing and BERT which is Bidirectional Encoder Representations from Transformers by applying this method it transfers learning with real social-media data. The goal is to help the model pick up on hidden hostility and subtle emotional cues, like sarcasm or passive aggression, that ordinary systems usually miss.

Expected Outcome

At the end, I’ll prepare a full research report that explains the development process, how the model works, its performance. My hope is that this project can contribute to building safer online spaces, where technology can actually recognize and stop harmful conversations before they cause more emotional and personal damage.

2. Introduction
Online social media platforms have become central to communication, but they have also enabled new avenues for harassment, known as cyberbullying. This behavior can have severe and lasting psychological impacts on victims. The technical challenge lies in language itself. The line between harmless "in-group" joking, sarcasm, and genuine malicious intent is often blurry. A statement like "You're a terrible person" could be a joke between friends or a targeted attack for some people with lots of sentiment and emotions . so from this project investigates the use of a state-of-the-art natural language processing (NLP) model, BERT, to tackle this problem. Our goal is to build and evaluate a multi-class classifier that can not only identify cyberbullying but also distinguish between different types, providing a more granular analysis.

3. Related Work
  3.1 A significant body of research exists on automated cyberbullying detection. In this project builds upon this, moving from traditional machine learning models to more advanced transformer-based architectures.
      Traditional Machine Learning (Baselines): Early approaches to this problem treated it as a standard text classification task. Researchers like Dadvar et al. (2013) and Nahar et al. (2014) used features like       TF-IDF (Term Frequency-Inverse Document Frequency) and n-grams to represent text. These features were then fed into classifiers like Support Vector Machines (SVM), Naive Bayes, and Logistic Regression.
      Key Finding: These models are very effective at identifying explicit, keyword-driven harassment (e.g., tweets with clear slurs or profanity).
      Key Limitation: They completely fail when context, sarcasm, or nuance is involved. They have no understanding of semantics.

4. Technical Background
    The tool used in this project is transfer learning via the BERT model.

    BERT (Bidirectional Encoder Representations from Transformers): BERT is a deep learning model developed by Google. Unlike older models that read text left-to-right (like an LSTM), BERT reads the entire            sequence of words at once. This "bidirectional" nature allows it to learn deep contextual relationships. 
    For example, it can understand that the word bank means something different in "river bank" vs. "money bank" based on the words around it.

5. Method
  The methodology follows a standard supervised machine learning pipeline.

  a. Dataset: I have downloafed the data set from kaggel cyberbullying_tweets.csv dataset, which contains 9,996 rows. Each row has a tweet_text and its corresponding cyberbullying_type.

  b. Labels: The six target classes are: age, ethnicity, gender, religion, other_cyberbullying, and not_cyberbullying. These are encoded into numerical labels (0-5) for the model.

  c. Model: The bert-base-uncased model was chosen as the base. This is a 12-layer Transformer model with 110 million parameters.

  d. Data Splitting: The data was split into a training set (7,996 samples, 80%) and a validation set (2,000 samples, 20%) to test the model's performance on unseen data.

  e. Preprocessing: Tokenization: The BertTokenizer converts raw text into a format BERT understands. This includes splitting words into sub-words (e.g., "bullying" -> "bulli" + "##ng"), adding special tokens          like [CLS] (start) and [SEP] (end), and converting tokens to numerical IDs.


6. Implementation
The solution was implemented in Python using Google Colab, leveraging its free GPU access.

Core Libraries: Pandas for data handling, Transformers (by Hugging Face) for the BERT model and tokenizer, PyTorch as the deep learning framework, and Scikit-learn for label encoding and evaluation metrics.

Data Pipeline: A custom CyberBullyingDataset class was created in PyTorch. This class handles the tokenization of one tweet at a time. This is fed into DataLoader, which batches the data (BATCH_SIZE = 16) and shuffles it for effective training.

  6.1 Training Process:
    
   a. Model: BertForSequenceClassification was loaded, pre-trained, and configured for 6 output labels.
    
   b. Optimizer: AdamW was used, which is the standard optimizer for BERT.
    
   c. Learning Rate (LR): A small LR of 2e-5 was chosen, as recommended for fine-tuning.
   
   d. Scheduler: A get_linear_schedule_with_warmup was used to adjust the LR during training, which helps model stability.
   
   e. Epochs: The model was trained for 3 full epochs. After epochs is done the model is saved in the path '/content/bert_cyberbullying_model'
