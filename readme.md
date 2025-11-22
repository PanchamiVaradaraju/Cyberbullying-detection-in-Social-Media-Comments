# Cyberbullying-detection-in-Social-Media-Comments
A machine learning project to detect cyberbullying in social media comments

1. Abstract

Cyberbullying Detection in Social Media Comments

Problem Statement 

Cyberbullying has slowly turned into one of the biggest problems of our generation. Every day, people are getting hurt by words shared online, and the emotional damage keeps growing as social media becomes a bigger part of our lives. What makes it worse is that the systems made to detect this kind of abuse still don’t really understand how humans communicate. They look for harsh words or harsh language, but most of the time, the real pain comes from messages that sound normal but carry a hidden meaning. Because of this, two frustrating things happen all the time normal posts get flagged for no reason, while truly hurtful comments pass through without any warning. It shows that these systems are not thinking the way people do. What we really need is a detection method that can feel and interpret context something more human-aware.

Proposed Solution

My idea is to create a smarter machine learning model that can sense the tone, mood, and intent behind a message, not just read the words on the screen. I’ll be using a strong pre trained language model like NLP which is Natural Language Processing and BERT which is Bidirectional Encoder Representations from Transformers by applying this method it transfers learning with real social-media data. The goal is to help the model pick up on hidden hostility and subtle emotional cues, like sarcasm or passive aggression, that ordinary systems usually miss.

Expected Outcome

At the end, I will prepare a full research report that explains the development process, how the model works, its performance. My hope is that this project can contribute to building safer online spaces, where technology can actually recognize and stop harmful conversations before they cause more emotional and personal damage.

2. Introduction

Online social media platforms have become central to communication, but they have also enabled new platform for harassment, known as cyberbullying. This behavior can have severe and lasting psychological impacts on victims. The technical challenge lies in language itself. The line between harmless in-group joking, sarcasm, and genuine malicious intent is often blurry. A statement like You're a terrible person could be a joke between friends but it will be a targeted attack for some people with lots of sentiment and emotions. so from this project investigates the use of a state-of-the-art natural language processing NLP model, BERT, to tackle this problem. The goal of this project is to build and evaluate a multi-class classifier that can not only identify cyberbullying but also distinguish between different types, providing a more granular analysis.

3. Related Work

3.1 A significant body of research exists on automated cyberbullying detection. 

In this project builds upon this, moving from traditional machine learning models to more advanced transformer-based architectures.Traditional Machine Learning (Baselines): Early approaches to this problem treated it as a standard text classification task. Researchers like Dadvar et al. (2013) and Nahar et al. (2014) used features like TF-IDF (Term Frequency-Inverse Document Frequency) and n-grams to represent text. These features were then fed into classifiers like Support Vector Machines (SVM), Naive Bayes, and Logistic Regression.

Model Used: These models are very effective at identifying explicit, keyword-driven harassment 
Limitation of this method: They completely fail when context, sarcasm, or nuance is involved. They have no understanding of semantics.

3.2 Early Deep Learning (LSTMs & CNNs): With the rise of deep learning, researchers began using Recurrent Neural Networks (RNNs), specifically LSTMs (Long Short-Term Memory), and Convolutional Neural Networks (CNNs). These models were able. to learn from the sequence of words, which was an improvement.

Model Used: Models by Pitsilis et al. (2018) showed that LSTMs could outperform traditional models by capturing some short-term contextual clues.
Limitation of this method: Their understanding of context is still shallow (often unidirectional) and they struggle with long-range dependencies in text.

4. Technical Background

   The tool used in this project is transfer learning via the BERT model.
    •	BERT (Bidirectional Encoder Representations from Transformers): BERT is a deep learning   model developed by Google. Unlike older models that read text left-to-right (like an LSTM), BERT reads the entire          sequence of words at once. This bidirectional nature allows it to learn deep contextual relationships. 
        For example, it can understand that the word bank means something different in river bank vs. money bank based on the words around it.
   

In this project, the 'bert-base-uncased' model was selected due to its balance of computational efficiency and performance. The dataset, obtained from Kaggle, contains 9,996 labeled tweets categorized into six classes. Data preprocessing, tokenization, and fine-tuning were executed using the Hugging Face Transformers library and PyTorch framework.


6. Method

   The methodology follows a standard supervised machine learning pipeline.
   a.	Dataset: I have downloaded the data set from kaggel cyberbullying_tweets.csv dataset, which contains 9,996 rows. Each row has a tweet text and its corresponding cyberbullying type.
   b.	Labels: The six target classes are: age, ethnicity, gender, religion, other_cyberbullying, and not_cyberbullying. These are encoded into numerical labels 0 to 5 for the model.
   c.	Model: The Bert-base-uncased model was chosen as the base. This is a 12-layer Transformer model with 110 million parameters.
   d.	Data Splitting: The data was split into a training set 7,996 samples, 80% and a validation set (2,000 samples, 20%) to test the model's performance on unseen data.
   e.	Preprocessing: Tokenization: The Bert Tokenizer converts raw text into a format BERT understands. This includes splitting words into sub-words (e.g., "bullying" -> "bulli" + "##ng"), adding special tokens         like [CLS] (start) and [SEP] (end), and converting tokens to numerical IDs.


7. Implementation

   The solution was implemented in Python using Google Colab, utilizing its free GPU access.

   Core Libraries: Pandas for data handling, Transformers (by Hugging Face) for the BERT model and tokenizer, PyTorch as the deep learning framework, and Scikit-learn for label encoding and evaluation metrics.

   Data Pipeline: A custom CyberBullyingDataset class was created in PyTorch. This class handles the tokenization of one tweet at a time. This is fed into DataLoader, which batches the data (BATCH_SIZE = 16) and     shuffles it for effective training.

  6.1 Training Process: 
  
  a.	Model: Bert for Sequence Classification was loaded, pre-trained, and configured for 6 output labels.
  
  b.	Optimizer: AdamW was used, which is the standard optimizer for BERT.
    
  c.	Scheduler: A get_linear_schedule_with_warmup was used to adjust the LR during training, which helps model stability.
  
  d.	Epochs: The model was trained for 3 full epochs. After epochs is done the model is saved in the path '/content/bert_cyberbullying_model'

7. Risk Analysis

The risk analysis evaluated in this project by using model's implementation as evidence which are technical, and project-related risks, using the direct results from the project


a.	TechnicalModel Overfitting: The model learns the training data well and doesn't generalize to new, unseen tweets.

Ex.  The model is static. It was trained on a single CSV file and saved. It has no connection to live data. 

b.	Project-related risks / Dataset Bias: The cyberbullying_tweets.csv dataset is not representative of all bullying or all forms of English.

Ex. The model is trained only on the data loaded from cyberbullying_tweets.csv. The labels are limited to the 6 classes encoded in cell 18. So if we give other data or different statement the model won’t identify so this is the main drawback

Conclusion 
This project enhanced my understanding of advanced NLP concepts, particularly transformer models. Through experimentation with BERT, I gained hands-on experience in fine-tuning, optimization, and ethical evaluation of AI systems.

The iterative nature of model improvement also reinforced project management and debugging skills. Future improvements would include expanding dataset diversity and deploying the model as a real-time web service.

In conclusion, this project demonstrates the feasibility of using BERT for cyberbullying detection. While challenges remain, particularly in contextual understanding and ethical deployment, the foundation established here can inform more robust and fair NLP-based moderation tools. Future work may involve integrating visual cues (images, emojis) and cross-lingual models to enhance detection accuracy.




