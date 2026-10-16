import os
import time
from typing import Optional

from groq import Groq

from src.llm.llm_provider import LLMProvider


class GroqProvider(LLMProvider):
    """
    Groq LLM Provider for APMOA.

    Responsibilities:
        - Connect to Groq API
        - Send system/user prompts
        - Request structured JSON output
        - Enforce request timeout
        - Retry transient failures
        - Provide useful execution logs
        - Return plain text to the Reasoning Agent
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_completion_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens

        # -----------------------------------------------------
        # Load API key
        # -----------------------------------------------------

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        # -----------------------------------------------------
        # Initialize Groq client
        # -----------------------------------------------------

        self.client = Groq(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=0,
        )

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an LLM response using Groq.

        The provider handles communication with the LLM.
        The Reasoning Agent remains responsible for
        process-mining reasoning and response interpretation.
        """

        # -----------------------------------------------------
        # Validate prompt
        # -----------------------------------------------------

        if not prompt or not prompt.strip():
            raise ValueError(
                "GroqProvider received an empty prompt."
            )

        # -----------------------------------------------------
        # Build messages
        # -----------------------------------------------------

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # -----------------------------------------------------
        # Debug information
        # -----------------------------------------------------

        prompt_length = len(prompt)

        system_length = (
            len(system_prompt)
            if system_prompt
            else 0
        )

        total_length = (
            prompt_length + system_length
        )

        print(
            "\n"
            + "-" * 60
        )

        print("GROQ API REQUEST")

        print("-" * 60)

        print(
            f"Model              : {self.model}"
        )

        print(
            f"Temperature        : {self.temperature}"
        )

        print(
            f"Timeout            : {self.timeout}s"
        )

        print(
            f"Max output tokens  : "
            f"{self.max_completion_tokens}"
        )

        print(
            f"JSON mode          : enabled"
        )

        print(
            f"Prompt chars       : "
            f"{prompt_length:,}"
        )

        print(
            f"System chars       : "
            f"{system_length:,}"
        )

        print(
            f"Total chars        : "
            f"{total_length:,}"
        )

        print(
            "Sending request to Groq..."
        )

        # -----------------------------------------------------
        # API request with retry
        # -----------------------------------------------------

        last_error = None

        for attempt in range(
            1,
            self.max_retries + 2,
        ):

            start_time = time.time()

            try:

                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,

                        # -------------------------------------------------
                        # IMPORTANT:
                        # Ask Groq for a JSON object instead of free-form
                        # text.
                        # -------------------------------------------------
                        response_format={
                            "type": "json_object"
                        },

                        # -------------------------------------------------
                        # Give the model enough room to finish the JSON.
                        # -------------------------------------------------
                        max_completion_tokens=(
                            self.max_completion_tokens
                        ),
                    )
                )

                elapsed = (
                    time.time() - start_time
                )

                # -------------------------------------------------
                # Validate response
                # -------------------------------------------------

                if not response.choices:

                    raise RuntimeError(
                        "Groq returned no choices."
                    )

                message = (
                    response
                    .choices[0]
                    .message
                )

                content = message.content

                if not content:

                    raise RuntimeError(
                        "Groq returned an empty response."
                    )

                # -------------------------------------------------
                # Inspect finish reason
                # -------------------------------------------------

                finish_reason = getattr(
                    response.choices[0],
                    "finish_reason",
                    None,
                )

                print(
                    f"Groq response received in "
                    f"{elapsed:.2f} seconds."
                )

                print(
                    f"Response chars : "
                    f"{len(content):,}"
                )

                print(
                    f"Finish reason  : "
                    f"{finish_reason}"
                )

                # -------------------------------------------------
                # Warn if generation was truncated
                # -------------------------------------------------

                if finish_reason in (
                    "length",
                    "max_tokens",
                ):

                    print(
                        "WARNING: Groq response may have "
                        "been truncated."
                    )

                print(
                    "-" * 60
                )

                return content

            except Exception as exc:

                elapsed = (
                    time.time() - start_time
                )

                last_error = exc

                print(
                    "\n"
                    + "-" * 60
                )

                print(
                    "GROQ REQUEST FAILED"
                )

                print(
                    "-" * 60
                )

                print(
                    f"Attempt : {attempt}"
                )

                print(
                    f"Time    : {elapsed:.2f}s"
                )

                print(
                    f"Error   : "
                    f"{type(exc).__name__}: {exc}"
                )

                # ---------------------------------------------
                # Retry if attempts remain
                # ---------------------------------------------

                if attempt <= self.max_retries:

                    wait_time = 2 * attempt

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                else:

                    print(
                        "Maximum Groq retry attempts reached."
                    )

        # -----------------------------------------------------
        # All attempts failed
        # -----------------------------------------------------

        raise RuntimeError(
            "Groq API request failed after "
            f"{self.max_retries + 1} attempts: "
            f"{last_error}"
        )