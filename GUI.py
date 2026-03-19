#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stress_trigger_test.py

GUI-based multiple-choice stress induction task for EEG experiments.

Purpose:
- Present timed multiple-choice questions
- Induce mild cognitive stress through time pressure and performance pressure
- Optionally send event markers through LSL for EEG synchronization

Requirements:
    pip install psychopy pylsl

Optional:
    If pylsl is not installed, the GUI still runs without event markers.
"""

import random
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

from psychopy import visual, core, event, gui

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Question:
    prompt: str
    choices: List[str]
    correct_index: int
    difficulty: str = "medium"


# ----------------------------
# LSL Marker Sender
# ----------------------------
class MarkerSender:
    def __init__(self, stream_name: str = "StressTaskMarkers"):
        self.outlet = None
        if LSL_AVAILABLE:
            info = StreamInfo(
                name=stream_name,
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id="stress_trigger_gui_001"
            )
            self.outlet = StreamOutlet(info)
            logging.info("LSL marker outlet created.")
        else:
            logging.warning("pylsl not installed. Running without LSL markers.")

    def push(self, marker: str) -> None:
        if self.outlet is not None:
            self.outlet.push_sample([marker])
        logging.info(f"Marker: {marker}")


# ----------------------------
# Stress Task GUI
# ----------------------------
class StressTriggerTask:
    def __init__(
        self,
        questions: List[Question],
        question_time_limit: float = 8.0,
        inter_trial_interval: float = 1.5,
        full_screen: bool = False
    ):
        self.questions = questions
        self.question_time_limit = question_time_limit
        self.inter_trial_interval = inter_trial_interval
        self.marker_sender = MarkerSender()

        self.win = visual.Window(
            size=(1280, 720),
            fullscr=full_screen,
            color="black",
            units="height"
        )

        self.clock = core.Clock()
        self.results = []

        self.title_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0.35),
            height=0.05,
            color="white",
            wrapWidth=1.5,
            bold=True
        )

        self.prompt_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0.18),
            height=0.04,
            color="white",
            wrapWidth=1.6
        )

        self.choice_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, -0.02),
            height=0.035,
            color="white",
            wrapWidth=1.6,
            alignText="left"
        )

        self.timer_text = visual.TextStim(
            self.win,
            text="",
            pos=(0.55, 0.4),
            height=0.04,
            color="red"
        )

        self.feedback_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, -0.32),
            height=0.04,
            color="orange",
            wrapWidth=1.4
        )

        self.pressure_text = visual.TextStim(
            self.win,
            text="Work quickly and accurately.",
            pos=(0, -0.42),
            height=0.03,
            color="lightgray"
        )

    def show_message(self, title: str, body: str, marker: Optional[str] = None):
        if marker:
            self.marker_sender.push(marker)

        self.title_text.text = title
        self.prompt_text.text = body
        self.choice_text.text = "Press SPACE to continue"
        self.timer_text.text = ""
        self.feedback_text.text = ""
        self.pressure_text.text = ""

        while True:
            self.title_text.draw()
            self.prompt_text.draw()
            self.choice_text.draw()
            self.win.flip()

            keys = event.getKeys()
            if "space" in keys:
                break
            if "escape" in keys:
                self.cleanup()
                raise SystemExit

    def draw_question(
        self,
        q: Question,
        question_number: int,
        total_questions: int,
        remaining_time: float,
        score: int
    ):
        self.title_text.text = f"Question {question_number}/{total_questions}"
        self.prompt_text.text = q.prompt

        choices_formatted = []
        for idx, choice in enumerate(q.choices):
            choices_formatted.append(f"{idx + 1}. {choice}")
        self.choice_text.text = "\n".join(choices_formatted)

        self.timer_text.text = f"{remaining_time:0.1f}s"
        self.feedback_text.text = f"Score: {score}"
        self.pressure_text.text = "You are being timed. Try not to fall behind."

        self.title_text.draw()
        self.prompt_text.draw()
        self.choice_text.draw()
        self.timer_text.draw()
        self.feedback_text.draw()
        self.pressure_text.draw()

    def run_trial(self, q: Question, trial_index: int, score: int):
        total_questions = len(self.questions)
        question_number = trial_index + 1

        self.marker_sender.push(f"QUESTION_START_{question_number}")
        self.marker_sender.push(f"DIFFICULTY_{q.difficulty.upper()}")

        trial_clock = core.Clock()
        response = None
        rt = None
        timed_out = False

        valid_keys = ["1", "2", "3", "4", "escape"]

        while trial_clock.getTime() < self.question_time_limit:
            remaining = self.question_time_limit - trial_clock.getTime()
            self.draw_question(q, question_number, total_questions, remaining, score)
            self.win.flip()

            keys = event.getKeys(timeStamped=trial_clock)
            for key, key_time in keys:
                if key == "escape":
                    self.cleanup()
                    raise SystemExit
                if key in ["1", "2", "3", "4"]:
                    response = int(key) - 1
                    rt = key_time
                    break

            if response is not None:
                break

        if response is None:
            timed_out = True
            self.marker_sender.push(f"QUESTION_TIMEOUT_{question_number}")
        else:
            self.marker_sender.push(f"RESPONSE_{question_number}_{response + 1}")

        correct = (response == q.correct_index) if response is not None else False

        if correct:
            self.marker_sender.push(f"CORRECT_{question_number}")
        elif timed_out:
            self.marker_sender.push(f"NO_RESPONSE_{question_number}")
        else:
            self.marker_sender.push(f"INCORRECT_{question_number}")

        trial_result = {
            "question_number": question_number,
            "prompt": q.prompt,
            "response_index": response,
            "correct_index": q.correct_index,
            "correct": correct,
            "rt_sec": rt,
            "timed_out": timed_out,
            "difficulty": q.difficulty
        }

        self.results.append(trial_result)

        # Short feedback to increase performance pressure
        if timed_out:
            feedback = "Too slow."
        elif correct:
            feedback = "Correct."
        else:
            feedback = "Incorrect."

        self.title_text.text = ""
        self.prompt_text.text = feedback
        self.choice_text.text = ""
        self.timer_text.text = ""
        self.feedback_text.text = ""
        self.pressure_text.text = ""

        self.prompt_text.draw()
        self.win.flip()
        core.wait(self.inter_trial_interval)

        return correct

    def run(self):
        score = 0

        self.show_message(
            title="Stress Task",
            body=(
                "You will answer multiple-choice questions under time pressure.\n\n"
                "Respond using keys 1, 2, 3, or 4.\n"
                "Each question must be answered quickly.\n\n"
                "Press SPACE to begin."
            ),
            marker="TASK_INSTRUCTIONS"
        )

        self.marker_sender.push("TASK_START")

        for i, q in enumerate(self.questions):
            if self.run_trial(q, i, score):
                score += 1

        self.marker_sender.push("TASK_END")

        self.show_message(
            title="Task Complete",
            body=f"You completed the task.\nFinal score: {score}/{len(self.questions)}",
            marker="TASK_COMPLETE"
        )

        self.cleanup()

    def cleanup(self):
        self.win.close()
        core.quit()


# ----------------------------
# Question bank
# ----------------------------
def build_question_bank() -> List[Question]:
    return [
        Question(
            prompt="If a student studies 3 subjects for 45 minutes each, how many total minutes do they study?",
            choices=["90", "120", "135", "150"],
            correct_index=2,
            difficulty="easy"
        ),
        Question(
            prompt="What is 17 × 6?",
            choices=["92", "98", "102", "108"],
            correct_index=2,
            difficulty="easy"
        ),
        Question(
            prompt="A class starts at 2:15 PM and ends at 3:50 PM. How long is the class?",
            choices=["1 hr 15 min", "1 hr 25 min", "1 hr 35 min", "1 hr 45 min"],
            correct_index=2,
            difficulty="medium"
        ),
        Question(
            prompt="Which number comes next in the sequence: 4, 9, 16, 25, ?",
            choices=["30", "35", "36", "49"],
            correct_index=2,
            difficulty="medium"
        ),
        Question(
            prompt="If 5 notebooks cost $17.50, what is the cost of 1 notebook?",
            choices=["$2.50", "$3.00", "$3.50", "$4.00"],
            correct_index=2,
            difficulty="medium"
        ),
        Question(
            prompt="A student scores 78, 84, and 90 on three tests. What is the average?",
            choices=["82", "84", "86", "88"],
            correct_index=1,
            difficulty="medium"
        ),
        Question(
            prompt="Which is the best synonym for 'concise'?",
            choices=["Detailed", "Brief", "Confusing", "Emotional"],
            correct_index=1,
            difficulty="easy"
        ),
        Question(
            prompt="What is 144 divided by 12?",
            choices=["10", "11", "12", "13"],
            correct_index=2,
            difficulty="easy"
        ),
        Question(
            prompt="If x + 7 = 19, what is x?",
            choices=["10", "11", "12", "13"],
            correct_index=2,
            difficulty="easy"
        ),
        Question(
            prompt="A task must be completed in 8 minutes. If 3 minutes have passed, how many remain?",
            choices=["3", "4", "5", "6"],
            correct_index=2,
            difficulty="easy"
        ),
    ]


# ----------------------------
# Main
# ----------------------------
def main():
    exp_info = {
        "Participant ID": "",
        "Time Limit (sec)": "8",
        "Shuffle Questions": True,
        "Fullscreen": False
    }

    dlg = gui.DlgFromDict(exp_info, title="Stress Trigger Task Setup")
    if not dlg.OK:
        return

    try:
        time_limit = float(exp_info["Time Limit (sec)"])
    except ValueError:
        logging.error("Invalid time limit entered.")
        return

    questions = build_question_bank()

    if exp_info["Shuffle Questions"]:
        random.shuffle(questions)

    task = StressTriggerTask(
        questions=questions,
        question_time_limit=time_limit,
        inter_trial_interval=1.0,
        full_screen=exp_info["Fullscreen"]
    )

    task.run()


if __name__ == "__main__":
    main()
