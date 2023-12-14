# LLM compression

The project demonstrates the work of LLM and the acceleration of inference using various optimization methods such as Graph Optimization, kv-caching.

To demonstrate the work, an application was made on Flask.
To run, you need to install the dependencies from the file requirements.txt and launch the application.

````
python -m pip install -r requirements.txt
flask --app app run
````

Two of the most effective methods were chosen - flash-attention-2 and kv-caching, they were used in the final application.
To demonstrate how the methods work, the ChatGLM2-6B neural network was selected and an application was written in Flask.

The work was done by Mikhail Zhelezin and Oksana Kozlova under the supervision of Professor Krylov Vladimir Vladimirovich.