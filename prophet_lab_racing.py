"""Prophet domain: horse racing. A drop-in file for prophet_lab.py.

Status: needs a data source. Prophet only uses free public data with no key and no sign-in, from sources that allow
automated download, and it does not read betting sites. A web search on 2026-10-24 found:
  - Hong Kong results: sold as paid exports by a third party.
  - Equibase (US): a one-off free dataset for calendar year 2023 offered as a promotion, terms not stated.
  - Japan: results only through paid scraping services.
  - France: the "Huggy Database" on Hugging Face; the announcement offers a free sample of one quarter of 1999 only, licence unconfirmed.
  - Betting-exchange price files are out of bounds for Prophet (no betting-site integrations).
When a legitimate open results file turns up, fill in the five calls below (see the plugin interface in RECORD.md) and
set status = "ready". Until then, hand-made calls still work: prophet_lab.py call --domain racing --question "..." --resolves YYYY-MM-DD --prob 0.3
"""
import prophet_lab as lab


class Adapter(lab.Adapter):
    name = "racing"
    title = "Horse racing: who wins the race"
    status = "needs_data_source"
    can_predict = False
    status_note = ("No free, keyless, openly downloadable results dataset was found (web search, 2026-10-24). Seen: Hong Kong results sold as paid exports; "
                   "an Equibase free dataset for calendar year 2023 only, as a promotion with terms not stated; Japanese results only through paid scrapers; "
                   "the French Huggy Database on Hugging Face with a free sample of one quarter (1999) and an unconfirmed licence. "
                   "Betting-exchange price files are excluded because Prophet does not use betting sites. "
                   "Hand-made calls still work: call --domain racing ... then resolve.")
    sources = []

    def fetch_history(self, lab_, fresh=False):
        raise lab.LabError("racing needs a data source: " + self.status_note)


ADAPTER = Adapter()
