#!/usr/bin/env python3
"""Generate PDF for DK vs DE gym products query."""

from fpdf import FPDF


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("seg", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Side {self.page_no()}", align="C")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_font("seg", "", "C:/Windows/Fonts/segoeui.ttf")
pdf.add_font("seg", "B", "C:/Windows/Fonts/segoeuib.ttf")
pdf.add_font("seg", "I", "C:/Windows/Fonts/segoeuii.ttf")
pdf.add_page()

SOURCE = "[K] "
OWN = "[E] "
BULL = "\u2022 "
DASH = "\u2014"


def title(text):
    pdf.set_font("seg", "B", 16)
    pdf.set_text_color(26, 26, 26)
    pdf.multi_cell(0, 8, text)
    pdf.ln(2)
    pdf.set_draw_color(50, 50, 50)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)


def h2(text):
    pdf.ln(3)
    pdf.set_font("seg", "B", 13)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 7, text)
    pdf.ln(2)


def h3(text):
    pdf.ln(2)
    pdf.set_font("seg", "B", 11)
    pdf.set_text_color(52, 73, 94)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)


def bullet(text, marker="", bold_prefix=""):
    icon = SOURCE if marker == "source" else OWN if marker == "own" else BULL
    pdf.set_font("seg", "", 10)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(5, 5.5, icon)
    if bold_prefix:
        pdf.set_font("seg", "B", 10)
        pdf.write(5.5, bold_prefix + " ")
        pdf.set_font("seg", "", 10)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)


def subbullet(text, marker=""):
    icon = SOURCE if marker == "source" else OWN if marker == "own" else "  "
    pdf.set_font("seg", "", 9.5)
    pdf.set_text_color(71, 71, 71)
    pdf.cell(8, 5.5, "")
    pdf.cell(5, 5.5, icon)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(0.5)


def separator():
    pdf.ln(3)
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


# ── Content ──────────────────────────────────────────────────────────────────

title("Danmark vs. Tyskland: Annoncering af\ngym-produkter (cuffs og straps)")

pdf.set_font("seg", "I", 10)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 5.5,
    "Sp\u00f8rgsm\u00e5l: Er der forskel p\u00e5 at reklamere for det samme produkt "
    "(cuffs og straps til gym) i Danmark og Tyskland?")
pdf.ln(4)

pdf.set_font("seg", "B", 9)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 5, "Forklaring:")
pdf.ln(5)
pdf.set_font("seg", "", 9)
pdf.cell(0, 5, "[K] = Direkte fra kilderne (Marketingpod)")
pdf.ln(5)
pdf.cell(0, 5, "[E] = Egne anbefalinger tilpasset gym-produkter")
pdf.ln(6)

separator()

# ── Kort svar ────────────────────────────────────────────────────────────────

h2("Kort svar")
bullet(
    "Ja, der er forskelle \u2014 men de er prim\u00e6rt tekniske og strukturelle, "
    "ikke n\u00f8dvendigvis kreative. Den st\u00f8rste fejl er at behandle hvert land "
    "som en separat silo-kampagne. I stedet skal du udnytte l\u00e6ring p\u00e5 "
    "tv\u00e6rs af lande.",
    "source")

separator()

# ── De 3 vigtigste forskelle ─────────────────────────────────────────────────

h2("De 3 vigtigste forskelle")

h3("1. Volumen og konkurrence")
bullet(
    "Tyskland har typisk 4-7x h\u00f8jere s\u00f8gevolumen end Danmark for "
    "samme produkter, men lavere klikpriser i Google Ads.",
    "source")
bullet(
    "For gym cuffs/straps: langt flere potentielle kunder i DE, "
    "paradoksalt nok billigere per klik.",
    "own")
bullet(
    "Tjek s\u00f8getrenden \u2014 er produktet stigende eller faldende i hvert marked? "
    "Et produkt kan v\u00e6re p\u00e5 vej ned i DK men stadig vokse i DE.",
    "source")

h3("2. Sprog og kreativt")
bullet(
    "Du beh\u00f8ver IKKE separate kampagner per land. "
    "Brug Metas dynamiske sprog-funktion:",
    "source")
subbullet("\u00c9n kampagne der d\u00e6kker b\u00e5de DK og DE", "source")
subbullet(
    "Annoncen p\u00e5 engelsk som fallback + dansk og tysk version "
    "under \"dynamisk sprog\"",
    "source")
subbullet(
    "Meta viser automatisk den rigtige sprogversion baseret p\u00e5 "
    "brugerens indstillinger",
    "source")
bullet(
    "Case: 14 lande \u2014 11 ud af 14 fik v\u00e6sentligt lavere CPA ved at "
    "samle dem i \u00e9n kampagne med dynamisk sprog.",
    "source")
bullet(
    "For gym-produkter: Kreativet (video af \u00f8velser, UGC) kan genbruges "
    "direkte \u2014 det er universelt. Kun tekst/CTA beh\u00f8ver overs\u00e6ttelse.",
    "own")

h3("3. Teknisk setup (pixel og katalog)")
bullet(
    "Brug samme pixel p\u00e5 begge lande. Separate pixels splitter din data "
    "og g\u00f8r l\u00e6ringsfasen l\u00e6ngere.",
    "source")
bullet(
    "Brug et Cross Border Business (CBB) katalog med dynamiske sproglag:",
    "source")
subbullet(
    "\u00c9t katalog med danske priser/navne + tyske priser/navne",
    "source")
subbullet(
    "DPA k\u00f8rer fra \u00e9t katalog til begge lande",
    "source")
subbullet(
    "Kun 50 konverteringer/uge for at komme ud af l\u00e6ringsfasen "
    "(vs. 100 med to kampagner)",
    "source")
bullet(
    "Dit st\u00e6rkeste land (DK) \"donerer\" l\u00e6ring til det nye land (DE), "
    "s\u00e5 du f\u00e5r bedre performance fra dag 1.",
    "source")

separator()

# ── Anbefalet setup ──────────────────────────────────────────────────────────

h2("Anbefalet setup for gym cuffs/straps i DK + DE")

h3("Meta/Facebook Ads")
bullet("1 pixel (global, p\u00e5 begge shops)", "source")
bullet("1 CBB-katalog med dynamiske sproglag (DK priser + DE priser)", "source")
bullet(
    "1 DPA-kampagne med dynamisk sprog (dansk + tysk + engelsk fallback)",
    "source")
bullet("1 prospecting-kampagne med dynamisk sprog", "source")
pdf.ln(1)
bullet(
    "UGC-video af folk der tr\u00e6ner med produktet \u2014 "
    "virker i begge lande uden \u00e6ndringer",
    "own")
bullet(
    "\"Grimme annoncer\" (selfie-stil, autentisk) performer bedre "
    "end poleret content",
    "source")
bullet(
    "Opret en personlig side som afsender (f.eks. \"Marcus fra [Brand]\") "
    "i stedet for virksomhedssiden",
    "source")

h3("Google Ads")
bullet(
    "DK: Byd p\u00e5 \"gym cuffs\", \"ankle straps tr\u00e6ning\", "
    "\"cable attachments\"",
    "own")
bullet(
    "DE: Byd p\u00e5 \"Fu\u00dfmanschetten Gym\", \"Kabelzug Griffe\", "
    "\"Gym Straps\"",
    "own")
bullet(
    "Klikpriserne er typisk lavere i DE trods h\u00f8jere volumen",
    "source")

separator()

# ── Ikke en forskel ──────────────────────────────────────────────────────────

h2("Hvad er det IKKE en forskel p\u00e5?")
bullet("M\u00e5lgruppen er den samme (fitness-entusiaster, 18-35)", "own")
bullet(
    "Produktet er universelt \u2014 ingen kulturelle forskelle i brug",
    "own")
bullet("Kreativt content (tr\u00e6ningsvideoer) virker cross-border", "own")
bullet("Prisforventningen er sammenlignelig", "own")
bullet(
    "Den prim\u00e6re forskel er teknisk/strukturel \u2014 ikke kreativ. "
    "S\u00e6t strukturen rigtigt op, og lad Meta optimere p\u00e5 tv\u00e6rs.",
    "source")

separator()

# ── Kilder ───────────────────────────────────────────────────────────────────

h2("Kilder (Marketingpod episoder)")
pdf.set_font("seg", "", 9)
pdf.set_text_color(80, 80, 80)
sources = [
    ("Webshop-sommer: Skalering af Facebook Ads til udlandet",
     "CBB-katalog, dynamisk sprog, pixel-strategi"),
    ("S\u00f8geordsanalyser til e-commerce ekspansion",
     "Volumen/klikpris-forskelle DK vs. DE"),
    ("De 9 bedste content-formater til h\u00f8j ROAS",
     "Content-strategi for e-commerce"),
    ("S\u00e5dan laver du nemt grimme annoncer der tjener penge",
     "Autentisk annonce-content"),
    ("Det nemmeste trick til bedre annonce-performance",
     "Personlig side som afsender"),
]
for ep, rel in sources:
    pdf.set_font("seg", "B", 9)
    pdf.cell(5, 5, BULL)
    pdf.write(5, ep)
    pdf.set_font("seg", "", 9)
    pdf.write(5, f" {DASH} {rel}")
    pdf.ln(6)

pdf.ln(6)
pdf.set_font("seg", "I", 8)
pdf.set_text_color(128)
pdf.cell(0, 5,
    "Genereret fra 288 indexerede episoder af Marketingpod (Obsidian Digital)",
    align="C")

pdf.output("queries/dk_vs_de_gym_produkter.pdf")
print("PDF saved: queries/dk_vs_de_gym_produkter.pdf")
