#!/usr/bin/env python3
"""Generate PDF for vinduespudser_strategi."""

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
    pdf.set_font("seg", "B", 18)
    pdf.set_text_color(26, 26, 26)
    pdf.multi_cell(0, 9, text)
    pdf.ln(2)
    pdf.set_draw_color(50, 50, 50)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)


def h2(text):
    pdf.ln(4)
    pdf.set_font("seg", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 8, text)
    pdf.ln(2)


def h3(text):
    pdf.ln(2)
    pdf.set_font("seg", "B", 11)
    pdf.set_text_color(52, 73, 94)
    pdf.multi_cell(0, 7, text)
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
    pdf.set_font("seg", "", 10)
    pdf.set_text_color(71, 71, 71)
    pdf.cell(10, 5.5, "")
    pdf.cell(5, 5.5, icon)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(0.5)


def note(text):
    pdf.set_font("seg", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(15, 5.5, "")
    pdf.multi_cell(0, 5, text)
    pdf.ln(1)


def separator():
    pdf.ln(3)
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


# ── Content ──────────────────────────────────────────────────────────────────

title("Vinduespudser: Online annoncering, kanaler og salgsstrategi")

pdf.set_font("seg", "I", 10)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 5.5,
    "Sp\u00f8rgsm\u00e5l: Jeg er vinduespudser. Hvilke kanaler skal jeg g\u00e5 p\u00e5? "
    "Hvilken strategi skal jeg have? Hvordan skal jeg s\u00e6lge mine services?")
pdf.ln(4)

# Legend
pdf.set_font("seg", "B", 9)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 5, "Forklaring:")
pdf.ln(5)
pdf.set_font("seg", "", 9)
pdf.cell(0, 5, "[K] = Direkte fra kilderne (Marketingpod)")
pdf.ln(5)
pdf.cell(0, 5, "[E] = Egne anbefalinger tilpasset vinduespudser-konteksten")
pdf.ln(6)

separator()

# ── Kanaler ──────────────────────────────────────────────────────────────────

h2("Kanal-prioritering (i r\u00e6kkef\u00f8lge)")

h3("1. Google Ads (h\u00f8jeste prioritet)")
bullet(f'Folk s\u00f8ger aktivt "vinduespudser [by]" {DASH} du fanger dem i det \u00f8jeblik, de har behov', "own")
bullet("Lokale s\u00f8gekampagner med geografisk targeting til dit omr\u00e5de", "own")
bullet("Billigt i forhold til mange andre brancher, fordi konkurrencen er lav", "own")

h3("2. Meta/Facebook Ads (lead-generator)")
bullet("Brug Lead Ads med betinget logik:", "source")
subbullet('Stil kvalificerende sp\u00f8rgsm\u00e5l: "Hvor mange vinduer har du?", "Privat eller erhverv?", "Postnummer?"', "own")
subbullet("Det filtrerer d\u00e5rlige leads og l\u00e6rer Facebooks algoritme at finde de rigtige kunder", "source")
subbullet("Dyrere per lead, men langt bedre kvalitet", "source")
bullet(f"Facebook er ikke dyrere end nogensinde {DASH} barrieren er lav, og Advantage+ g\u00f8r det nemt selv for begyndere", "source")

h3("3. Organisk/lokalt (gratis)")
bullet(f'Google Business Profile {DASH} kritisk for "vinduespudser n\u00e6r mig" s\u00f8gninger', "own")
bullet(f"Facebook Lives / TikTok fra jobs {DASH} vis f\u00f8r/efter af vinduer, det koster intet og giver r\u00e6kkevidde", "source")

separator()

# ── Strategi ─────────────────────────────────────────────────────────────────

h2("Strategi: Lign ikke en annonce")
bullet('Opret en Facebook-side i dit eget navn: "Thomas fra [Firmanavn]" eller "Thomas Vinduespudsning"', "source")
bullet("K\u00f8r annoncer fra den personlige side, ikke virksomhedssiden", "source")
bullet("Brug autentisk content: billeder fra jobs med din telefon, f\u00f8r/efter, korte videoer", "source")
bullet("Det performer markant bedre fordi det ligner oprigtigt indhold, ikke reklame", "source")

separator()

# ── Salg ─────────────────────────────────────────────────────────────────────

h2("S\u00e5dan s\u00e6lger du dine services")

h3(f"Kortsigtet {DASH} f\u00e5 kunder i n\u00e6ste uge")
bullet("Skriv direkte til relevante kontakter (LinkedIn eller e-mail). Segmenteret, personligt, ikke spam.", "source", f"Struktureret outreach {DASH}")
note("For vinduespudsere: ejendomsadministrationer, boligforeninger, ejendomsm\u00e6glere.")
bullet("Facebook Lead Ad med et simpelt tilbud.", "source")
note('F.eks.: "Gratis pr\u00f8vevask af 3 vinduer" eller "F\u00f8rste pudsning 50% rabat"')
bullet('"Jeg pudser vinduer hos din nabo, vil du have et tilbud?"', "own", f"D\u00f8r-til-d\u00f8r i nabolag {DASH}")

h3(f"Langsigtet {DASH} stabil kundestr\u00f8m")
bullet("S\u00e6lg faste pudsninger (hver 4./6./8. uge). Det giver forudsigelig oms\u00e6tning.", "own", f"Abonnementsmodel {DASH}")
bullet("Send automatisk p\u00e5mindelser og s\u00e6sonkampagner.", "source", f"E-mail flows {DASH}")
note('F.eks. reminder til n\u00e6ste pudsning, for\u00e5r = "se verden klart igen"')
bullet("Bed tilfredse kunder filme et kort klip af resultatet. Brug det som annoncemateriale.", "source", f"Anmeldelser/UGC {DASH}")

separator()

# ── Budget ───────────────────────────────────────────────────────────────────

h2("Budget-anbefaling")
bullet("Start med 3.000-5.000 kr./md. fordelt: ca. 60% Google Ads (aktiv eftersp\u00f8rgsel), ca. 40% Facebook Lead Ads (ny eftersp\u00f8rgsel)", "own")
bullet("Skal\u00e9r det der virker. Hvis Google leverer leads til 100 kr. der konverterer, put mere der.", "own")

separator()

# ── Kilder ───────────────────────────────────────────────────────────────────

h2("Kilder (Marketingpod episoder)")
pdf.set_font("seg", "", 9)
pdf.set_text_color(80, 80, 80)
sources = [
    ("Glem branding, s\u00e5dan f\u00e5r du kunder NU", "Kortsigtede tiltag til kundegenerering"),
    ("Hacket til bedre kvalitet p\u00e5 Facebook Lead Ads", "Betinget logik i Lead Ads"),
    ("Det nemmeste trick til bedre annonce-performance", "Personlig side som afsender"),
    ("Er Facebook annoncering... billigt?", "CPM-priser og Advantage+"),
    ("S\u00e5dan skaber du salg i B2B her og nu", "Struktureret outreach"),
    ("Har du lavet de 7 vigtigste e-mail flows?", "Automatiserede e-mail flows"),
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
pdf.cell(0, 5, "Genereret fra 288 indexerede episoder af Marketingpod (Obsidian Digital)", align="C")

pdf.output("queries/vinduespudser_strategi.pdf")
print("PDF saved: queries/vinduespudser_strategi.pdf")
