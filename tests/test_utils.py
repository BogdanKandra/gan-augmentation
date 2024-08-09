from logging import Logger

import pytest

import scripts.utils as utils


class TestUtils:
    """ Tests for the utils script """
    def test_get_logger(self):
        """ Tests that the correct logger instance is returned by the get_logger function """
        logger = utils.get_logger(__name__)
        assert type(logger) == Logger
        assert logger.getEffectiveLevel() == 10

    def test_is_perfect_square_wrong_type(self):
        """ Tests that a TypeError is raised when the type of the parameter passed to is_perfect_square is not int """
        with pytest.raises(TypeError) as e:
            utils.is_perfect_square("XVI")
            assert e == 'Unexpected type for parameter "number" (Expected <int>, given <str>)'

    def test_is_perfect_square_integers(self):
        """ Tests that the is_perfect_square function correctly identifies perfect squares up to 100 """
        expected = [False] * 101
        for i in [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]:
            expected[i] = True

        for i in range(101):
            assert utils.is_perfect_square(i) == expected[i]
