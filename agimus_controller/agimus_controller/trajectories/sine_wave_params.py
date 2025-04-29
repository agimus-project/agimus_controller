import numpy as np


class SinWaveParams:
    """
    Class to store parameters for a sine wave trajectory.
    """

    def __init__(
        self, amplitude: list[float], period: list[float], scale_duration: list[float]
    ):
        """
        Initialize the sine wave parameters.

        :param amplitude: Amplitude of the sine wave.
        :param period: Period of the sine wave.
        :scale_duration: Duration from 0 velocity to the sin wave velocity.
        """
        self.amplitude = amplitude
        self.period = period
        self.scale_duration = scale_duration

    @property
    def pulsation(self) -> float:
        """
        Calculate the pulsation of the sine wave.

        :return: Pulsation of the sine wave.
        """
        return (2 * np.pi * np.array(self.frequency)).tolist()

    @property
    def frequency(self) -> float:
        """
        Calculate the frequency of the sine wave.

        :return: Frequency of the sine wave.
        """
        safe_array = np.where(np.abs(self.period) < 1e-6, np.nan, self.period)
        inverted_array = 1.0 / safe_array
        return np.nan_to_num(inverted_array, nan=0.0).tolist()
