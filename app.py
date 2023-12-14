from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from datetime import datetime
from neural_network import model, model2

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


def generate_answer1(question):
    return model.run_model(question)

def generate_answer2(question):
    return model2.run_model(question)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        question_type = request.form.get("type")
        question = request.form["question"]

        # Выбор правильной истории на основе типа запроса
        history_key = "history1" if question_type == "question1" else "history2"

        if history_key not in session:
            session[history_key] = []
        session[history_key].append(question)

        start_time = datetime.now()
        
        # Выбор функции генерации ответа
        if question_type == "question1":
            answer = generate_answer1(question)
        elif question_type == "question2":
            answer = generate_answer2(question)
        else:
            answer = "Неизвестный тип вопроса"

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        return jsonify({"answer": answer, "response_time": response_time})
    else:
        return render_template("question.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)