from config.config import *
from agent.agent import Agent
from abc import ABC, abstractmethod
from prompt.task_description import *
from agent.responder import *


class StateBase(ABC):
    def __init__(self, task, guide=None):
        self.task = task
        self.guide = None

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
        super().__init__(task, guide)

    def enter(self):
        self.task.step = -1

    def exec(self):
        return Guide(self.task)


class Guide(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task, guide)
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id]
        }

    def enter(self):
        pass

    def exec(self):
        # TODO: Need to query LLM to get the guiding questions, and set the guiding questions here
        agent = Agent(prompt=StateBase.read_prompt("guide"), **self.prompt_variables)
        guiding_questions = agent.generate()["guide"]
        print("These are the guiding questions: ")
        for i in guiding_questions:
            print(i)
        return Search(self.task, prompt=None, guide=self.guide)


class Search(StateBase):
    def __init__(self, task, prompt=None, guide=None):
        super().__init__(task, guide)
        self.prompt = prompt
        self.model = None
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": self.task.get_history_gq(self.task.step).strip(),
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
        query = agent.generate()["query"]

        responder = Responder(query)
        response = responder.generate()  # You may want to store this

        self.task.generate_task.append(
            {
                "step": self.task.step,
                "query": query,
                "thought": thought,
                "response": response,
            }
        )

        return Stop(self.task)


class Stop(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task, guide)
        self.model = None
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": self.task.get_history_sc(self.task.step).strip(),
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(prompt=StateBase.read_prompt("stop"), **self.prompt_variables)
        results = agent.generate()

        if "Terminate" in results["action"]:
            return Finish(self.task)
        else:
            return Search(self.task)


class Finish(StateBase):
    def __init__(self, task, guide=None):
        super().__init__(task, guide)

    def enter(self):
        pass

    def exec(self):
        pass
