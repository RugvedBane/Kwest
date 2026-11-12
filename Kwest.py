# imports
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from main import get_competition_info, get_leaderboard, get_my_submissions
from memory import save_experiment, get_memory_context
load_dotenv()


# calling groq model
def create_model():
    model_name = os.getenv("GROQ_MODEL")

    return ChatGroq(model=model_name, temperature=0.2, max_tokens=4096)

# built system prompt 
#  - system message (the kwest persona + competition context)
#  - placeholder for chat history
#  - human message

# to built system first we need to built the context
def build_context(comp_slug):

    comp_info = get_competition_info(comp_slug)
    leaderboard = get_leaderboard(comp_slug)
    submissions = get_my_submissions(comp_slug)
    experiment_memory = get_memory_context(comp_slug)

    competiton_context = f"""
    COMPETITION INFO:
    {comp_info}

    LEADERBOARD:
    {leaderboard}
    
    MY SUBMISSIONS:
    {submissions}

    EXPERIMENT MEMORY:
    {experiment_memory}
    
    """

    return competiton_context

def build_prompt():
    system_message = (
        "You are Kwest, a personalized Kaggle agent for its user."
        "Your job is to carefully analyze the user’s competition submissions, leaderboard rank and competition info, and suggest specific improvements to increase their rank." 
        "Never give generic advice."
        "If you have incomplete information ask the user, never assume anything."
        "Never suggest an experiment already tried unless you propose a meaningful modification."
        "Here is the competition data: {competition_context}"
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", "{question}")
    ])


# built the chain using | 
def built_chain(prompt, model):
    chain = prompt | model

    return chain


# main function
# ask user for competition slug
# fetch context
# create model
# build prompt
# build chain
# empty chat history list
# while loop:
#     get user input
#     call chain
#     print response
#     append to chat history

def main():
    comp_slug = input("Enter competition slug(e.g titanic): ")
    competition_context = build_context(comp_slug)
    model = create_model()
    prompt = build_prompt()
    chain = built_chain(prompt, model)
    chat_history = []

    while True:
        question = input('user: ')

        if question.lower() in ['exit', 'quit']:
            break

        if question.lower().startswith('log:'):
            parts = question[4:].strip().split("|")
            save_experiment(comp_slug, parts[0], float(parts[1]), [parts[2]])
            competition_context = build_context(comp_slug)
            continue

        response = chain.invoke({
            "competition_context": competition_context,
            "chat_history": chat_history,
            "question": question
        })

        print("\nKwest: ", response.content)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response.content))
        chat_history = chat_history[-6:]


if __name__ == "__main__":
    main()