#!/usr/bin/env python3

"""Scrapes the catalog from the Siemens site."""

import argparse
import os.path

import requests
import lxml.html


def scrape(args: argparse.Namespace) -> None:
    """Scrapes the catalog from the Siemens site.

    I didn't want to do this, because it is bad manners, but they forced me to.
    Initially I planned to use the Excel download option on their site, but as it turned
    out, it is implemented with boundless incompetency.  It lets you download only 500
    items at a time, which is too few for the whole catalog of MCBs, which numbers about
    2250 items.  But even if you set filters to reduce the number of selected items to
    less than 500, the download option will not honour the filters, and you will always
    end up with the same first 500 elements of 2250, and with no way to get at the other
    1750.
    """

    url = "https://mall.industry.siemens.com/mall/Catalog/GetProducts/ProductsOrAccessories"
    params = {
        "treeName": "CatalogTree",
        "IsAccessories": False,
    }
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Cache-control": "no-cache",
        "Content-type": "application/json; charset=UTF-8",
        "Pragma": "no-cache",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Cookie": "RegionUrl=/de",
    }

    nodes = {
        10256170: "LS-Schalter",
        10263940: "FI-Schutzschalter",
        10263941: "FI/LS-Schalter",
        10263942: "FI-Blöcke",
        10411002: "Brandschutzschalter",
        10046634: "Klemmen 8WH",
        10263947: "Überspannungsschutzgeräte",
    }

    for nodeId in sorted(nodes.keys()):
        params["nodeId"] = nodeId
        print(nodes[nodeId])
        for i in range(1, 250):
            params["page"] = i
            r = requests.post(url, json=params, headers=headers)
            print(i, end=" ", flush=True)
            html = r.json()["HtmlContent"]
            tree = lxml.html.fromstring(html)
            tds: list[lxml.html._Element] = tree.xpath(  # type: ignore
                "//td[@class='ProductName']"  # trust me I'm an engineer
            )
            if len(tds) == 0:
                break
            filename = os.path.join(args.output, f"{nodeId}-{i:03}.txt")
            with open(filename, "w") as fp:
                for td in tds:
                    children = list(td)
                    article = children[0].text_content().strip()
                    desc = children[1].text_content().strip()
                    fp.write(f"{article} {desc}\n")
        print("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the commandline parser"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,  # don't wrap my description
        description=__doc__,
    )

    parser.add_argument("-v", action="store_true")

    parser.add_argument(
        "output",
        metavar="DIR",
        help="Put the files into DIR.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if os.path.isdir(args.output):
        scrape(args)
    else:
        print("Error: The output directory must exist!")
        parser.print_usage()
