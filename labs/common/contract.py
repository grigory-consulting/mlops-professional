"""Data Contract für Kundenanfragen (Lab 1, Demo 4; Kapitel 3).

- `check_contract`: die Regeln der Folie "Data Contract prüfen" als Funktion,
  eine Spalte je Regel, eine Zeile je Anfrage (True = Regel verletzt).
  Zusätzlich eine Variante der Ziffern-Probe, die Leer- und Bindestriche
  vorher entfernt (Kartennummern stehen oft in Vierergruppen).
- `known_word_share`: einfache Sprachprobe ohne Zusatzpaket. Anteil der Wörter
  einer Anfrage, die der Tokenizer des Encoders als ganzes Wort kennt. Das
  Vokabular von `minilm-l6` und `bge-small` ist englisch; deutsche Wörter
  zerfallen in Wortstücke.
- `NEW_DELIVERY`: eine Lieferung mit typischen Problemfällen, von Hand
  geschrieben (Banking77 selbst ist bereinigt und enthält keine).
- `GERMAN_QUERIES`: 20 Anfragen aus dem Eval-Set des Kurses mit deutscher
  Übersetzung (von Hand), um zu messen, was die Regel `language: en` schützt.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MAX_CHARS = 500
MIN_DIGITS = 8
SEPARATORS = r"[\s\-]"
_WORD = re.compile(r"[a-zäöüß]+")


def check_contract(df: pd.DataFrame, intents: list[str], min_digits: int = MIN_DIGITS) -> pd.DataFrame:
    """Prüft eine Lieferung (Spalten text, intent) gegen die Regeln des Contracts.

    Die ersten fünf Spalten sind die Regeln der Folie in Kapitel 3, die letzte
    ist die Variante der Ziffern-Probe ohne Trennzeichen.
    """
    text = df["text"]
    joined = text.str.replace(SEPARATORS, "", regex=True)
    return pd.DataFrame({
        "leerer Text": text.str.strip() == "",
        f"mehr als {MAX_CHARS} Zeichen": text.str.len() > MAX_CHARS,
        "Zeichen außerhalb ASCII": ~text.map(str.isascii),
        "unbekanntes Anliegen": ~df["intent"].isin(intents),
        f"Ziffernfolge ab {min_digits} Stellen": text.str.contains(rf"\d{{{min_digits},}}"),
        "dieselbe ohne Trennzeichen": joined.str.contains(rf"\d{{{min_digits},}}"),
    }, index=df.index)


def known_word_share(texts: list[str], tokenizer) -> np.ndarray:
    """Anteil der Wörter je Text, die im Vokabular des Tokenizers als ganzes Wort stehen.

    Wörter sind Buchstabenfolgen (Zahlen und Satzzeichen zählen nicht). Ein Text
    ohne Wörter bekommt 1,0: Leere Texte fängt eine eigene Regel ab.
    """
    vocab = tokenizer.get_vocab()
    shares = []
    for text in texts:
        words = _WORD.findall(text.lower())
        shares.append(np.mean([word in vocab for word in words]) if words else 1.0)
    return np.asarray(shares, dtype=float)


# Eine neue Lieferung des Kundenservice: Text, Anliegen und was an der Zeile besonders ist.
NEW_DELIVERY = pd.DataFrame([
    ("My card still hasn't arrived after two weeks.", "card_arrival", "normale Anfrage"),
    ("   ", "card_arrival", "leer (nur Leerzeichen)"),
    ("Meine Karte ist noch nicht angekommen.", "card_arrival", "deutsch (Karte)"),
    ("Ich möchte eine Rückerstattung für meinen Einkauf.", "request_refund", "deutsch (Erstattung)"),
    ("Please block card 4111 1111 1111 1111, it was stolen.", "lost_or_stolen_card", "Kartennummer"),
    ("Send the refund to GB29NWBK60161331926819 please.", "request_refund", "IBAN"),
    ("Can I get a mortgage for my flat?", "mortgage", "neues Anliegen"),
    ("Dear customer service team, I am writing to you because I ordered a new card more than three weeks "
     "ago and it has still not arrived. I already contacted you twice by phone and both times I was told "
     "that it would arrive within a few days, but nothing has happened since then. In the meantime I cannot "
     "pay in shops, I cannot withdraw cash and I had to borrow money from a friend to pay my rent. Could you "
     "please check where the card is, tell me when it will finally arrive and whether you can send a "
     "replacement by express delivery? Thank you very much, kind regards.", "card_arrival", "E-Mail statt Anfrage"),
], columns=["text", "intent", "fall"])


# (Anfrage aus dem Eval-Set, Anliegen, deutsche Übersetzung von Hand)
GERMAN_QUERIES = [
    ("What is the source of my money?", "verify_source_of_funds", "Was ist die Herkunft meines Geldes?"),
    ("I want a refund for my purchase", "request_refund", "Ich möchte für meinen Einkauf eine Rückerstattung."),
    ("How much do I have to pay for the exchange fee?", "exchange_charge",
     "Wie viel muss ich für die Umtauschgebühr bezahlen?"),
    ("What do I need to do to activate my new card?", "activate_my_card",
     "Was muss ich tun, um meine neue Karte zu aktivieren?"),
    ("When I topped up, the app reverted my funds.", "top_up_reverted",
     "Als ich aufgeladen habe, hat die App mein Geld zurückgebucht."),
    ("How long can an EU transfer take?", "pending_transfer", "Wie lange kann eine EU-Überweisung dauern?"),
    ("How do I exchange my GBP to AUD?", "exchange_via_app", "Wie tausche ich meine GBP in AUD um?"),
    ("Why hasn't my card been delivered?", "card_arrival", "Warum wurde meine Karte nicht zugestellt?"),
    ("What will I be charged for a physical card?", "order_physical_card", "Was kostet mich eine physische Karte?"),
    ("Can I request a physical card?", "order_physical_card", "Kann ich eine physische Karte beantragen?"),
    ("I have been charged a fee for paying with card", "card_payment_fee_charged",
     "Mir wurde eine Gebühr für die Kartenzahlung berechnet."),
    ("My passcode doesn't seem to be working", "passcode_forgotten", "Mein Passcode scheint nicht zu funktionieren."),
    ("What is this €1 fee in my statement?", "extra_charge_on_statement",
     "Was ist diese Gebühr von 1 € auf meinem Kontoauszug?"),
    ("Why is my card not working?", "card_not_working", "Warum funktioniert meine Karte nicht?"),
    ("Is using my card free?", "card_payment_fee_charged", "Ist die Nutzung meiner Karte kostenlos?"),
    ("I think my child used my card while I wasn't home.", "compromised_card",
     "Ich glaube, mein Kind hat meine Karte benutzt, während ich nicht zu Hause war."),
    ("Can you tell me what countries you operate in?", "country_support", "In welchen Ländern sind Sie tätig?"),
    ("What cards and currencies do you support?", "supported_cards_and_currencies",
     "Welche Karten und Währungen unterstützen Sie?"),
    ("What is stopping my money from transferring?", "failed_transfer", "Warum wird mein Geld nicht überwiesen?"),
    ("How would I go about getting a second card?", "getting_spare_card", "Wie bekomme ich eine zweite Karte?"),
]
