from config.config import *
from agent.agent import Agent
from abc import ABC, abstractmethod
from prompt.task_description import *
from agent.responder import *


class StateBase(ABC):
    def __init__(self, task, guide=None, conversation=None):
        self.task = task
        self.guide = guide
        self.conversation = conversation

    @abstractmethod
    def enter(self):
        pass

    @abstractmethod
    def exec(self):
        pass

    @staticmethod
    def read_prompt(prompt_type):
        prompt_file = os.path.join(ROOT_PATH, "prompt", f"{prompt_type}.txt")
        assert os.path.exists(prompt_file)

        with open(prompt_file, "r", encoding="utf8") as file:
            prompt = file.read()
        return prompt


class Init(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task)

    def enter(self):
        self.task.step = -1

    def exec(self):
        return Guide(self.task)


class Guide(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task)
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id]
        }

    def enter(self):
        pass

    def exec(self):
        # TODO: Need to query LLM to get the guiding questions, and set the guiding questions here
        agent = Agent(prompt=StateBase.read_prompt("guide"), **self.prompt_variables)
        guiding_questions = agent.generate()["guide"]
        self.guide = guiding_questions
        print("These are the guiding questions: ")
        for i in guiding_questions:
            print(i)
        return Search(self.task, query=None, guide=self.guide)


class Search(StateBase):
    def __init__(self, task, query=None, guide=None, history=None):
        super().__init__(task, guide)
        self.query = query
        self.model = None
        self.history = history
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": history,
            "guiding_questions": self.guide,
        }

    def enter(self):
        self.task.step += 1

    def get_thought(self):
        agent = Agent(prompt=StateBase.read_prompt("thought"), **self.prompt_variables)
        thought = agent.generate()["thought"]
        return thought

    def exec(self):
        thought = self.get_thought()

        agent = Agent(
            prompt=StateBase.read_prompt("query"),
            thought=thought,
            **self.prompt_variables,
        )
        query = agent.generate()["query"] if self.query is None else self.query

        responder = Responder(query)
        response = responder.generate()  # You may want to store this
        query_response = {"query": query, "response": response}
        self.history = [query_response] if self.history is None else self.history.append(query_response)

        self.task.generate_task.append(
            {
                "step": self.task.step,
                "query": query,
                "thought": thought,
                "response": response,
            }
        )

        return Stop(self.task, self.guide, self.history)


class Stop(StateBase):
    def __init__(self, task, guide=None, history = None):
        super().__init__(task, guide)
        self.model = None
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": history,
            "guiding_questions": self.guide,
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(prompt=StateBase.read_prompt("stop"), **self.prompt_variables)
        results = agent.generate()

        if "Terminate" in results["action"]:
            return Finish(self.task)
        else:
            return Search(self.task, results["follow-up"], self.history)


class Finish(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task, guide)

    def enter(self):
        pass

    def exec(self):
        pass
