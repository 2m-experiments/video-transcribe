# Queries

Derived outputs — answers and summaries generated **from** the transcribed groups
(the RAG/analysis end of the pipeline). These are *not* sources; for where the source
videos come from, see [`../channel/`](../channel/README.md).

Each doc records its own source citations inline. This index maps every doc to the
**group** it draws on.

| Doc | Type | Source group | Drawn from |
|-----|------|--------------|-----------|
| `group4_masterclass_takeaways.md` | Summary — "Top 10 Takeaways, Content Masterclass 2026" | **group4** | Obsidian webinar transcript (`transcriptions/group4/…`) |
| `dk_vs_de_gym_produkter.md` (+ `.pdf`) | Q&A — advertising gym products (cuffs/straps) DK vs DE | **group3** | 3 Marketingpod episodes (cited in doc) |
| `vinduespudser_strategi.md` (+ `.pdf`) | Q&A — window-cleaner channels, strategy & sales | **group3** | 3 Marketingpod episodes (cited in doc) |

## Conventions

- Markdown is the primary artifact; `.pdf` is an exported copy of the same doc.
- Inside the marketing Q&A docs, points marked **📎** come straight from the source
  episodes; **💡** are added recommendations adapted to the asker's context.
- Source episodes are listed under a **"Kilder"** section at the bottom of each doc.

> Note: despite living together here, these don't all belong to one group — only the
> masterclass takeaways come from group4; the two marketing Q&As come from group3.
