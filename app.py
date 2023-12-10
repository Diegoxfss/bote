from flask import Flask, render_template, request, redirect, url_for
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

answers_dir_path = os.path.dirname(os.path.abspath(__file__))
answers_file_path = os.path.join(answers_dir_path, "answers.txt")
answers = {}
conversation = []

def read_answers_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    values = parts[1].split(",")
                    answers[key] = [value.strip() for value in values]

if os.path.isfile(answers_file_path):
    read_answers_file(answers_file_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', conversation=conversation)

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['user_query']

    if user_query.lower() == "exit":
        return render_template('result.html', output="Exiting the program.")

    if not answers:
        return render_template('result.html', output="No questions available. Please add questions to your answers.txt file.")
    else:
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(list(answers.keys()) + [user_query])
        similarity_scores = cosine_similarity(question_vectors[:-1], question_vectors[-1])
        most_similar_index = similarity_scores.argmax()
        most_similar_question = list(answers.keys())[most_similar_index]

        compatible_answers = answers[most_similar_question]
        bot_answer = compatible_answers[0]

        if len(compatible_answers) > 1:
            max_similarity = similarity_scores[most_similar_index].max()
            best_answer_index = similarity_scores[most_similar_index].argmax()
            bot_answer = compatible_answers[best_answer_index]

        # Update the conversation history
        conversation.append({'user': 'User', 'user_question': user_query, 'bot': 'Diego-Bot', 'bot_answer': bot_answer})

        return redirect(url_for('index'))

@app.route('/add_answer', methods=['GET'])
def add_answer_form():
    return render_template('add_answer.html')

@app.route('/add_answer', methods=['POST'])
def add_answer():
    question = request.form['question']
    answer = request.form['answer']

    with open(answers_file_path, "a") as file:
        file.write(f"{question.strip()} : {answer.strip()}\n")

    # Update the answers dictionary
    answers[question.strip()] = [answer.strip()]

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)


