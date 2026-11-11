(() => {
    // Verified publication metadata. Keys and formatting are normalized for reuse.
    // Each entry records its source so future updates can be checked against it.
    const citations = {
        // https://arxiv.org/bibtex/2609.32827
        // https://arxiv.org/abs/2609.32827
        "mapl": String.raw`@misc{liu2026improvingllmcollaborationmultiagent,
  title = {Improving {LLM} Collaboration via Multi-Agent Preference Learning},
  author = {Shuo Liu and Xinzichen Li and Tianle Chen and Christopher Amato},
  year = {2026},
  eprint = {2609.32827},
  archiveprefix = {arXiv},
  primaryclass = {cs.AI},
  url = {https://arxiv.org/abs/2609.32827},
}`,
        // https://arxiv.org/bibtex/2605.06595
        // https://arxiv.org/abs/2605.06595
        "crona": String.raw`@misc{liu2026crossmodalnavigationmultiagentreinforcement,
  title = {Cross-Modal Navigation with Multi-Agent Reinforcement Learning},
  author = {Shuo Liu and Xinzichen Li and Christopher Amato},
  year = {2026},
  eprint = {2605.06595},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
  url = {https://arxiv.org/abs/2605.06595},
}`,
        // https://neurips.cc/virtual/2026/loc/sydney/161964
        // https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PTA
        // https://arxiv.org/pdf/2610.35398
        // Formatted from official NeurIPS/OpenReview metadata; per-paper export unavailable.
        "bcmaac": String.raw`@inproceedings{chen2026belief,
  title = {Belief-Conditioned Multi-Agent Actor Critic for Decentralized {LLM} Collaboration},
  author = {Chen, Tianle and Liu, Shuo and Amato, Christopher},
  booktitle = {From Pretrained Representations to Acting Agents, {NeurIPS} 2026 Workshop},
  year = {2026},
  url = {https://openreview.net/forum?id=az4kbwdYyi}
}`,
        // https://rlj.org/papers/baisero2026fixing
        // https://rlj.org/volumes/7
        // https://rlj.cs.umass.edu/2026/papers/Paper121.html
        // Official RLJ export; the publisher has not yet assigned pages or a DOI.
        "value-decomposition": String.raw`@article{baisero2026fixing,
  title = {Fixing Incomplete Value Function Decomposition for Multi-Agent Reinforcement Learning},
  author = {Andrea Baisero and Rupali Bhati and Shuo Liu and Aathira Sunil Pillai and Christopher Amato},
  journal = {Reinforcement Learning Journal},
  volume = {7},
  year = {2026},
  url = {https://rlj.org/papers/baisero2026fixing}
}`,
        // https://proceedings.mlr.press/v306/liu26k.html
        "collm": String.raw`@inproceedings{pmlr-v306-liu26k,
  title = {Learning Decentralized {LLM} Collaboration with Multi-Agent Actor Critic},
  author = {Liu, Shuo and Chen, Tianle and Amiri, Ryan and Amato, Christopher},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages = {75603--75623},
  year = {2026},
  editor = {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = {306},
  series = {Proceedings of Machine Learning Research},
  month = {06--11 Jul},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v306/liu26k.html}
}`,
        // https://ojs.aaai.org/index.php/AAAI/article/view/40487
        // https://api.crossref.org/works/10.1609/aaai.v40i38.40487/transform/application/x-bibtex
        "magrpo": String.raw`@article{liu2026collaboration,
  title = {{LLM} Collaboration with Multi-Agent Reinforcement Learning},
  volume = {40},
  url = {http://dx.doi.org/10.1609/aaai.v40i38.40487},
  doi = {10.1609/aaai.v40i38.40487},
  number = {38},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  author = {Liu, Shuo and Liang, Zeyu and Lyu, Xueguang and Amato, Christopher},
  year = {2026},
  pages = {32150--32158},
  month = {Mar}
}`,
        // https://api.crossref.org/works/10.3390/g15020012/transform/application/x-bibtex
        // https://www.mdpi.com/2073-4336/15/2/12
        "mfg-pigno": String.raw`@article{chen2024physics,
  title = {Physics-Informed Graph Neural Operator for Mean Field Games on Graph: A Scalable Learning Approach},
  volume = {15},
  issn = {2073-4336},
  url = {https://doi.org/10.3390/g15020012},
  doi = {10.3390/g15020012},
  number = {2},
  journal = {Games},
  publisher = {MDPI AG},
  author = {Chen, Xu and Liu, Shuo and Di, Xuan},
  year = {2024},
  month = mar,
  pages = {12}
}`,
        // https://api.crossref.org/works/10.3390/math12060803/transform/application/x-bibtex
        // https://www.mdpi.com/2227-7390/12/6/803
        "mfg-pino": String.raw`@article{liu2024scalable,
  title = {Scalable Learning for Spatiotemporal Mean Field Games Using Physics-Informed Neural Operator},
  volume = {12},
  issn = {2227-7390},
  url = {https://doi.org/10.3390/math12060803},
  doi = {10.3390/math12060803},
  number = {6},
  journal = {Mathematics},
  publisher = {MDPI AG},
  author = {Liu, Shuo and Chen, Xu and Di, Xuan},
  year = {2024},
  month = mar,
  pages = {803}
}`,
        // https://api.crossref.org/works/10.3233/FAIA230299/transform/application/x-bibtex
        // https://ebooks.iospress.nl/doi/10.3233/FAIA230299
        // Crossref export with proceedings type and pagination verified by IOS Press.
        "dual-mfg": String.raw`@inproceedings{chen2023dual,
  title = {Learning Dual Mean Field Games on Graphs},
  isbn = {9781643684376},
  issn = {1879-8314},
  url = {https://doi.org/10.3233/FAIA230299},
  doi = {10.3233/FAIA230299},
  booktitle = {ECAI 2023},
  publisher = {IOS Press},
  author = {Chen, Xu and Liu, Shuo and Di, Xuan},
  year = {2023},
  month = sep,
  pages = {421--428},
  series = {Frontiers in Artificial Intelligence and Applications},
  volume = {372}
}`,
        // https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1079.pdf
        // https://www.ifaamas.org/Proceedings/aamas2023/forms/contents.htm
        // https://dblp1.uni-trier.de/rec/conf/atal/ChenLD23.html
        // Formatted from the IFAAMAS published paper and bibliographic record.
        "mfg-hybrid": String.raw`@inproceedings{chen2023hybrid,
  author = {Chen, Xu and Liu, Shuo and Di, Xuan},
  title = {A Hybrid Framework of Reinforcement Learning and Physics-Informed Deep Learning for Spatiotemporal Mean Field Games},
  booktitle = {Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems},
  year = {2023},
  pages = {1079--1087},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
  doi = {10.5555/3545946.3598748},
  url = {https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1079.pdf}
}`,
        // https://api.crossref.org/works/10.1109/ITSC55140.2022.9922047/transform/application/x-bibtex
        // https://dblp1.uni-trier.de/rec/conf/itsc/LiuWCFD22.html
        // https://par.nsf.gov/servlets/purl/10447569
        // https://github.com/LovelyBuggies/sumo-gym
        "smart-eflo": String.raw`@inproceedings{liu2022smarteflo,
  author = {Liu, Shuo and Wang, Yunhao and Chen, Xu and Fu, Yongjie and Di, Xuan},
  title = {{SMART-eFlo}: An Integrated {SUMO-Gym} Framework for Multi-Agent Reinforcement Learning in Electric Fleet Management Problem},
  booktitle = {2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)},
  year = {2022},
  pages = {3026--3031},
  publisher = {IEEE},
  doi = {10.1109/ITSC55140.2022.9922047},
  url = {https://doi.org/10.1109/ITSC55140.2022.9922047}
}`,
    };

    const dialog = document.createElement('dialog');
    dialog.className = 'citation-dialog';
    dialog.setAttribute('aria-labelledby', 'citation-heading');
    dialog.innerHTML = `
        <div class="citation-dialog-header">
            <h2 id="citation-heading">BibTeX</h2>
            <button type="button" autofocus aria-live="polite">Copy</button>
        </div>
        <pre></pre>
    `;
    document.body.append(dialog);

    const citationText = dialog.querySelector('pre');
    const copyButton = dialog.querySelector('button');

    function renderCitation(citation) {
        const lines = citation.split('\n');
        const fields = lines.map(line => /^\s*(\w+)\s*=\s*/.exec(line));
        const fieldWidth = Math.max(...fields.map(field => field?.[1].length ?? 0));
        // Two leading spaces and " = " separate each field from its value.
        citationText.style.setProperty('--value-indent', `${fieldWidth + 5}ch`);
        citationText.replaceChildren();
        lines.forEach((line, index) => {
            const row = document.createElement('span');
            const field = fields[index];
            if (field) {
                row.className = 'bibtex-field';
                row.textContent = `  ${field[1].padEnd(fieldWidth)} = ${line.slice(field[0].length)}`;
            } else {
                row.textContent = line;
            }
            if (index > 0) citationText.append('\n');
            citationText.append(row);
        });
    }

    copyButton.addEventListener('click', async () => {
        const firstLine = citationText.firstElementChild;
        let copied = false;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(citationText.textContent);
                copied = true;
            }
        } catch {
            // Use the visible text selection if the browser denies clipboard access.
        }
        if (!dialog.open || firstLine !== citationText.firstElementChild) return;

        if (!copied) {
            const selection = window.getSelection();
            selection.selectAllChildren(citationText);
            try {
                copied = document.execCommand('copy');
            } catch {
                // Leave the citation selected for manual copying.
            }
            if (copied) selection.removeAllRanges();
        }
        copyButton.textContent = copied ? 'Copied!' : 'Copy manually';
    });
    dialog.addEventListener('click', (event) => {
        const bounds = dialog.getBoundingClientRect();
        if (event.target === dialog && (
            event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom
        )) {
            dialog.close();
        }
    });

    document.querySelectorAll('[data-citation]').forEach((link) => {
        link.setAttribute('aria-haspopup', 'dialog');
        const title = link.closest('li').querySelector('paper').textContent.trim();
        link.setAttribute('aria-label', `Cite ${title}`);
        link.addEventListener('click', (event) => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
                return;
            }
            const citation = citations[link.dataset.citation];
            if (!citation || typeof dialog.showModal !== 'function') {
                return;
            }
            event.preventDefault();
            renderCitation(citation);
            copyButton.textContent = 'Copy';
            dialog.showModal();
            citationText.scrollTop = 0;
        });
    });
})();
