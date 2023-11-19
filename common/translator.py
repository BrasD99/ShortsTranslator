from googletrans import Translator
from common.mapper import map

class TextHelper:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, src_lang, dst_lang):
        output = self.translator.translate(
            text, src=src_lang, dest=map(dst_lang))
        return output.text