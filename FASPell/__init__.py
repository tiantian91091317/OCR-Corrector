# -*- coding: utf-8 -*-
# Brief: faspel.api

from .faspell import SpellChecker

spell_checker = SpellChecker()
correct = spell_checker.correction_service