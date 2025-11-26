import pandas as pd
import re
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Завантаження даних
print("=== ЗАВАНТАЖЕННЯ ДАНИХ ===")
# Використовуємо меншу вибірку для швидкості порівняння
LIMIT = 35000

yelp_path = '/home/pavlo/lab7/yelp_review_polarity_csv/'
train_df = pd.read_csv(yelp_path + 'train.csv', header=None, names=['label', 'text'], nrows=LIMIT)
test_df = pd.read_csv(yelp_path + 'test.csv', header=None, names=['label', 'text'], nrows=5000)

# Перекодування міток (1->0, 2->1)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 1 else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 1 else 1)

y_train = train_df['label'].values
y_test = test_df['label'].values

stop_words = set(stopwords.words('english'))


# Базова очистка (однакова для обох методів, щоб порівнювати саме токенізацію)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("Попередня очистка текстів...")
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# Параметри для обох моделей
VOCAB_SIZE = 10000
MAX_LENGTH = 100


# Функція створення моделі
def create_model(name):
    """Створює однакову архітектуру для чесного порівняння"""
    model = Sequential([
        Input(shape=(MAX_LENGTH,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=128),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name=name)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Метод 1 - Keras Tokenizer
print("\n--- МЕТОД 1: KERAS STANDARD TOKENIZER ---")

# Keras Tokenizer сам робить спліт по пробілах і видаляє пунктуацію
keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
keras_tokenizer.fit_on_texts(train_df['clean_text'])

X_train_keras = keras_tokenizer.texts_to_sequences(train_df['clean_text'])
X_test_keras = keras_tokenizer.texts_to_sequences(test_df['clean_text'])

X_train_pad_keras = pad_sequences(X_train_keras, maxlen=MAX_LENGTH, padding='post')
X_test_pad_keras = pad_sequences(X_test_keras, maxlen=MAX_LENGTH, padding='post')

model_keras = create_model("Keras_Model")
history_keras = model_keras.fit(
    X_train_pad_keras, y_train,
    epochs=3, batch_size=64,
    validation_data=(X_test_pad_keras, y_test),
    verbose=1
)

# Метод 2 - NLTK Tokenizer
print("\n--- МЕТОД 2: NLTK WORD TOKENIZE ---")
print("Токенізація через NLTK...")


def nltk_tokenize_text(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and word.isalnum()]


# Застосовуємо NLTK до кожного рядка
X_train_nltk_tokens = train_df['clean_text'].apply(nltk_tokenize_text).tolist()
X_test_nltk_tokens = test_df['clean_text'].apply(nltk_tokenize_text).tolist()

# Тепер нам треба перетворити слова в числа.
mapper = Tokenizer(num_words=VOCAB_SIZE)
mapper.fit_on_texts(X_train_nltk_tokens) # Приймає список списків

X_train_nltk_seq = mapper.texts_to_sequences(X_train_nltk_tokens)
X_test_nltk_seq = mapper.texts_to_sequences(X_test_nltk_tokens)

X_train_pad_nltk = pad_sequences(X_train_nltk_seq, maxlen=MAX_LENGTH, padding='post')
X_test_pad_nltk = pad_sequences(X_test_nltk_seq, maxlen=MAX_LENGTH, padding='post')

model_nltk = create_model("NLTK_Model")
history_nltk = model_nltk.fit(
    X_train_pad_nltk, y_train,
    epochs=3, batch_size=64,
    validation_data=(X_test_pad_nltk, y_test),
    verbose=1
)


# Порівняння результатів
def get_full_metrics(model, X_test_data, y_true, name):
    """Розраховує Accuracy, Precision, Recall, F1-Score для моделі."""
    # Отримуємо передбачення
    predictions = model.predict(X_test_data, verbose=0)
    # Конвертуємо ймовірності в бінарні класи (0 або 1)
    y_pred = (predictions > 0.5).astype(int)

    # Розраховуємо повний звіт
    # output_dict=True повертає результати у вигляді словника для легкої обробки
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Витягуємо ключові метрики. Для бінарної класифікації зазвичай беруть
    # метрики для Позитивного класу ('1')
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),  # Глобальна метрика
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1-Score': report['1']['f1-score']
    }
    return metrics


# Отримуємо метрики для Keras Tokenizer
metrics_keras = get_full_metrics(model_keras, X_test_pad_keras, y_test, "Keras")
# Отримуємо метрики для NLTK Tokenizer
metrics_nltk = get_full_metrics(model_nltk, X_test_pad_nltk, y_test, "NLTK")
# Об'єднуємо результати у DataFrame
results_df = pd.DataFrame([metrics_keras, metrics_nltk])
# Форматуємо дані для побудови гістограми
results_df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

print("\n=== ТАБЛИЦЯ ПОРІВНЯЛЬНИХ МЕТРИК ===")
print(results_df.set_index('Model'))

# Побудова порівняльної гістограми
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Metric',
    y='Score',
    hue='Model',
    data=results_df_melted,
    palette=['#3498db', '#e74c3c']
)

plt.title('Порівняння метрик: Keras vs NLTK Tokenization')
plt.ylabel('Оцінка (Score)')
plt.ylim(results_df_melted['Score'].min() * 0.95, 1.0)  # Динамічне обмеження осі Y
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Токенізатор')
plt.tight_layout()

# Збереження та відображення графіка
plt.savefig("output/tokenizers_metrics.png")
plt.show()
