from html import escape
from typing import List, Dict
from datetime import datetime

def render_html_table(
    rows: List[Dict],
    title: str = "Stock Signals",
    output_file: str = "./docs/index.html",
) -> str:
    """Render a list of dicts as a complete HTML page containing a styled single-row-per-dict table with inlined, minified CSS."""

    if not isinstance(rows, list):
        raise TypeError("rows must be a list of dicts")

    excluded = {"yahoo_details", "signal", "histogram", "last_close", "sma30_value"}

    # make clickable links for keys containing 'ticker'
    for r in rows:
        if "ticker" in r:
            ticker = r["ticker"]
            r["ticker"] = (
                f'<a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" rel="noopener noreferrer">{ticker}</a>'
            )

    # Collect headers
    headers = []
    for r in rows:
        if not isinstance(r, dict):
            raise TypeError("each row must be a dict")
        for k in r.keys():
            if k in excluded:
                continue
            if k not in headers:
                headers.append(k)

    def td(val: object, row: Dict) -> str:
        txt = "" if val is None else str(val)
        lower_txt = txt.lower()

        if "%" in txt:
            try:
                num = float(txt.replace("%", "").replace(",", ""))
                if num > 0:
                    return f"<td class='sma_pos'>{escape(txt)}</td>"
                elif num < 0:
                    return f"<td class='sma_neg'>{escape(txt)}</td>"
            except ValueError:
                pass

        elif "price" in row and "ycp" in row and val == row.get("price"):
            price, ycp = float(row["price"]), float(row["ycp"])
            if price > ycp:
                return f"<td class='pos_price'>{price}</td>"
            elif price < ycp:
                return f"<td class='neg_price'>{price}</td>"
        elif "name" in row and val == row.get("name"):
            return f"<td class='name'>{txt}</td>"

        elif "<a href" in lower_txt:
            return f"<td class='ticker'>{txt}</td>"

        # NEGATIVE
        elif any(word in lower_txt for word in [
            "no-entry","negative","strong-sell","poor","oversold","below",
            "decreasing","below-band","low"
        ]):
            return f"<td class='negative'>{escape(txt)}</td>"
        # SMALL NEGATIVE
        elif any(word in lower_txt for word in ["cautious-buy","sell","weak","avoid"]):
            return f"<td class='small_neg'>{escape(txt)}</td>"

        # NEUTRAL
        elif any(word in lower_txt for word in ["hold","neutral", "moderate"]):
            return f"<td class='neutral'>{escape(txt)}</td>"

        # SMALL POSITIVE
        elif any(word in lower_txt for word in ["buy","cautious-sell","good", "yes", "consider-keep", "normal","stable"]):
            return f"<td class='small_pos'>{escape(txt)}</td>"

        # POSITIVE
        elif any(word in lower_txt for word in [
            "positive","strong-buy", "don't-exit", "excellent","increasing","strong","good",
            "above","high","overbought"
        ]):
            return f"<td class='positive'>{escape(txt)}</td>"

        return f"<td>{txt}</td>"

    # Build rows
    rows_html = []
    for r in rows:
        cells = [td(r.get(h, ""), r) for h in headers]
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    header_html = "".join(f"<th>{escape(str(h))}</th>" for h in headers)

    # --- CSS block ---
    css = """
    body { font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif; padding:6px; background:linear-gradient(135deg,#f0f4f8,#d9e2ec);}
    .card { background:white; border-radius:6px; box-shadow:0 6px 10px rgba(0,0,0,0.1); padding:4px; max-width:85%; margin:1px; overflow-x:auto;}
    table { border-collapse:collapse; width:100%; border-radius:6px; overflow:hidden;}
    caption { text-align:center; font-size:1rem; padding-bottom:7px; font-weight:400; color:#243b53;}
    th,td { color:#333332; padding:5px 6px; border-bottom:1px solid #ced9e5; text-align:center; vertical-align:top; font-size:0.7rem;}
    th { background:#f8fafc; color:#334e68; font-weight:400; text-transform:uppercase;}
    tr:hover td { border-bottom: 1px solid #ee6c6c; border-top-color: 1px solid #ee6c6c; transition:background 0.5s ease;}
    .small { font-size:0.55rem; color:#52667a; margin-top:10px;}
    .positive { color:#fff; background-color:#709a8b;}
    .negative { color:#fff; background-color:#ee6c6c;}
    .small_pos { color:#fff; background-color:#9ccbba;}
    .small_neg { color:#fff; background-color:#ddac78;}
    .neutral { color:#35362c; background-color:#f6f6e5;}
    .ticker { color:#333332; text-align:center; vertical-align: middle; border-radius:16px; font-weight:400; padding:3px 4px; background-color:#f0f4f8;}
    .neg_price { color:#d64545;}
    .pos_price { color:#709a8b;}
    .sma_neg { color:#d64545;}
    .sma_pos { color:#709a8b;}
    .name { text-align:left; font-weight:600; color:#243b53;}
    .name:hover {cursor:default;}
    .positive:hover { background-color:#f1f1f1;}
    @media (max-width:768px){
      th,td{ padding:6px 2px; font-size:0.6rem;}
      caption{ font-size:1.2rem;}
      .small{ font-size:0.5rem;}
    }
    """

    def minify_css(css: str) -> str:
        return " ".join(line.strip() for line in css.splitlines() if line.strip())

    css_min = minify_css(css)

    # --- HTML page ---
    html_page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(title)}</title>
  <style>{css_min}</style>
</head>
<body>
  <div class="card">
    <table>
      <caption>{escape(title)}</caption>
      <thead>
        <tr>{header_html}</tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
    <p class="small">Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  </div>
</body>
</html>"""

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_page)

    return html_page

