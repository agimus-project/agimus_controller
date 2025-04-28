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
        return 2 * np.pi / self.period

    @property
    def frequency(self) -> float:
        """
        Calculate the frequency of the sine wave.

        :return: Frequency of the sine wave.
        """
        return 1.0 / self.period
