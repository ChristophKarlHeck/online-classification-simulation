# online_min_max.py
import math
import numpy as np
from typing import List

class OnlineWindow:
    """
    A class to maintain a fixed-size window of float values, supporting updates
    with new values and retrieval of the current minimum and maximum within the window.
    """

    def __init__(self, window_size: int):
        """
        Initializes a fixed-size window with the given size.
        The window is pre-filled with 0.0, and internal counters are initialized.
        
        :param window_size: The size of the window.
        """
        self.window: List[float] = [0.0] * window_size
        self.window_size = window_size
        self.current_index = 0  # Included for potential future use.
        self.count = 0

    def update(self, arr: List[float]) -> None:
        """
        Updates the window with new float values provided in the list `arr`.
        If there is room in the window (i.e. not full yet), new values are copied into the window.
        Once the window is full, the window shifts left and the last element of `arr` is appended.
        
        :param arr: List of new float values to update the window with.
        """
        if self.count < self.window_size - len(arr) + 1:
            # Insert new values into the window at the current count index.
            for i, value in enumerate(arr):
                self.window[self.count + i] = value
            self.count += 1
        else:
            # Shift the window left and insert the last value from arr at the end.
            self.window[:-1] = self.window[1:]
            self.window[-1] = arr[-1]
            self.count = self.window_size

    def get_max_value(self) -> float:
        """
        Returns the maximum value from the current window.
        If the window is not fully populated, only considers the valid portion.
        If no valid data exists, returns negative infinity.
        
        :return: The maximum float value in the window.
        """
        if self.count == 0:
            return -math.inf
        if self.count < self.window_size:
            return max(self.window[:self.count + 1])
        else:
            return max(self.window)

    def get_min_value(self) -> float:
        """
        Returns the minimum value from the current window.
        If the window is not fully populated, only considers the valid portion.
        If no valid data exists, returns positive infinity.
        
        :return: The minimum float value in the window.
        """
        if self.count == 0:
            return math.inf
        if self.count < self.window_size:
            return min(self.window[:self.count + 1])
        else:
            return min(self.window)

    def get_mean(self) -> float:
        """
        Returns the minimum value from the current window.
        If the window is not fully populated, only considers the valid portion.
        If no valid data exists, returns positive infinity.
        
        :return: The minimum float value in the window.
        """
        if self.count == 0:
            return math.inf
        if self.count < self.window_size:
            return np.mean(self.window[:self.count + 1])
        else:
            return np.mean(self.window)

    def get_std(self) -> float:
        """
        Returns the minimum value from the current window.
        If the window is not fully populated, only considers the valid portion.
        If no valid data exists, returns positive infinity.
        
        :return: The minimum float value in the window.
        """
        if self.count == 0:
            return math.inf
        if self.count < self.window_size:
            return np.std(self.window[:self.count + 1])
        else:
            return np.std(self.window)

# Optional: Module test code
# if __name__ == '__main__':
#     # Example usage:
#     omm = OnlineMinMax(5)
#     omm.update([1.0, 2.0])
#     print("Max:", omm.get_max_value())  # Expected: 2.0
#     print("Min:", omm.get_min_value())  # Expected: 1.0
#     omm.update([3.0])
#     omm.update([4.0])
#     omm.update([5.0])
#     omm.update([6.0])
#     print("Max:", omm.get_max_value())  # Should reflect the updated window values.
#     print("Min:", omm.get_min_value())
