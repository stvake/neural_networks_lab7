import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
stop_words = set(stopwords.words('english'))

print("Завантаження даних...")

# Завантажуємо CSV файли
# header=None, бо у файлі немає заголовків.
# names=['label', 'text'] дає назви колонкам.
# nrows=50000 - обмежуємо кількість даних для пришвидшення
yelp_path = '/home/pavlo/lab7/yelp_review_polarity_csv/'
train_df = pd.read_csv(yelp_path + 'train.csv', header=None, names=['label', 'text'], nrows=50000)
test_df = pd.read_csv(yelp_path + 'test.csv', header=None, names=['label', 'text'], nrows=10000)

print(f"Завантажено тренувальних записів: {len(train_df)}")
print(f"Завантажено тестових записів: {len(test_df)}")


# 2. ПОПЕРЕДНЯ ОБРОБКА
def preprocess_text(text):
    text = str(text).lower()  # Нижній регістр
    text = re.sub(r'\W', ' ', text)  # Видалення спецсимволів
    text = re.sub(r'\s+', ' ', text).strip()  # Видалення зайвих пробілів
    # Видалення стоп-слів
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


print("Очищення тексту...")
train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

# Перекодування міток:
# У датасеті: 1 = Негативний, 2 = Позитивний
# Для LSTM: 0 = Негативний, 1 = Позитивний
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 1 else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 1 else 1)

X_train = train_df['clean_text']
y_train = train_df['label'].values
X_test = test_df['clean_text']
y_test = test_df['label'].values

# 3. ТОКЕНІЗАЦІЯ ТА ПАДДІНГ
print("Токенізація...")

# Максимальна кількість слів у словнику
vocab_size = 10000
# Довжина, до якої будуть зрізані або доповнені вектори
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)  # Навчаємо токенізатор тільки на train

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# 4. АРХІТЕКТУРА LSTM
print("Побудова моделі LSTM...")

model = Sequential([
    Input(shape=(max_length,)),
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, return_sequences=True),  # Перший шар LSTM повертає послідовності для наступного
    Dropout(0.5),  # Регуляризація для запобігання перенавчанню
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid для бінарної класифікації (0 або 1)
])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. НАВЧАННЯ НЕЙРОМЕРЕЖІ
print("Початок навчання...")

history = model.fit(
    X_train_pad, y_train,
    epochs=7,
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    verbose=1
)

# Графік навчання
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точність моделі')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.savefig("output/training.png")
plt.show()

# 6. ОЦІНКА ТА МАТРИЦЯ ПОМИЛОК
# Отримуємо передбачення (ймовірності від 0 до 1)
predictions = model.predict(X_test_pad)
# Перетворюємо у класи (0 або 1) за порогом 0.5
y_pred = (predictions > 0.5).astype(int)

# Побудова матриці помилок
print("\nМатриця помилок:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nДетальна оцінка за класами:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 7. ТЕСТУВАННЯ НА ВЛАСНИХ ДАНИХ
def analyze_sentiment(input_text):
    # Попередня обробка
    cleaned = preprocess_text(input_text)
    # Токенізація
    seq = tokenizer.texts_to_sequences([cleaned])
    # Паддінг
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    # Передбачення
    prob = model.predict(padded)[0][0]

    sentiment = "Positive" if prob > 0.5 else "Negative"
    return sentiment, prob


print("\n--- Тестування на прикладах ---")

sample_text_1 = "The service was absolutely terrible and the food was cold."
sent_1, prob_1 = analyze_sentiment(sample_text_1)
print(f"Text: {sample_text_1}\nSentiment: {sent_1} (Probability: {prob_1:.4f})\n")

sample_text_2 = "Amazing experience! The staff was friendly and the atmosphere was great."
sent_2, prob_2 = analyze_sentiment(sample_text_2)
print(f"Text: {sample_text_2}\nSentiment: {sent_2} (Probability: {prob_2:.4f})\n")

sample_text_3 = "The LSTM neural network is very powerful for sequence processing."
sent_3, prob_3 = analyze_sentiment(sample_text_3)
print(f"Text: {sample_text_3}\nSentiment: {sent_3} (Probability: {prob_3:.4f})\n")