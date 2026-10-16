from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Generic interface for Large Language Model providers.

    Agents communicate through this interface instead of
    directly depending on a specific LLM provider.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate an LLM response.

        Parameters
        ----------
        prompt:
            User/task prompt.

        system_prompt:
            Optional system instruction.

        Returns
        -------
        str
            Generated response.
        """

        raise NotImplementedError