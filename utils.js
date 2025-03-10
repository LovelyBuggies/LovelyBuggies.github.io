function rlSortList(sortBy) {
    sortList('rl-post-list', sortBy);
}

function marlSortList(sortBy) {
    sortList('marl-post-list', sortBy);
}

function sortList(listId, sortBy) {
    const list = document.getElementById(listId);
    const items = Array.from(list.getElementsByTagName("li"));

    items.sort((a, b) => {
        if (sortBy === "latest") {
            const dateA = new Date(a.dataset.date);
            const dateB = new Date(b.dataset.date);
            return dateB - dateA;
        } else if (sortBy === "earliest") {
            const dateA = new Date(a.dataset.date);
            const dateB = new Date(b.dataset.date);
            return dateA - dateB;
        } else if (sortBy === "alphabetic") {
            const textA = a.querySelector("paper a").textContent.toLowerCase();
            const textB = b.querySelector("paper a").textContent.toLowerCase();
            return textA.localeCompare(textB);
        } else if (sortBy === "depth") {
            const depthA = parseInt(a.dataset.depth);
            const depthB = parseInt(b.dataset.depth);
            return depthA - depthB;
        }
    });

    list.innerHTML = "";
    items.forEach(item => list.appendChild(item));
}

let lastSelectedTag = null;
function showTaggedItems(tag) {
    const lists = ['rl-post-list', 'marl-post-list'];
    const filteredList = document.getElementById("filtered-list");
    const descriptionElementId = "tag-description";
    const parentElement = filteredList.parentElement;
    let descriptionElement = document.getElementById(descriptionElementId);
    if (descriptionElement) {
        descriptionElement.remove();
    }
    if (lastSelectedTag === tag) {
        filteredList.innerHTML = ""; 
        lastSelectedTag = null; 
        return;
    }
    lastSelectedTag = tag;
    filteredList.innerHTML = "";
    if (tag === "Planning" || tag === "Learning" || tag === "Decentralized" || tag === "Brainstorm") {
        descriptionElement = document.createElement("div");
        descriptionElement.id = descriptionElementId;
        descriptionElement.style.marginBottom = "10px";
        descriptionElement.style.fontStyle = "italic";
        descriptionElement.style.fontWeight = "bold";
        descriptionElement.style.fontSize = "80%";
        // descriptionElement.style.color = "#222222";
        // here  I manually use the same style as <ps> in the index.css
        if (tag === "Planning") {
            descriptionElement.textContent = "Planning is the process of deciding on a course of action to achieve specified goals.";
        } else if (tag === "Learning") {
            descriptionElement.textContent = "Learning is the process of improving performance based on experience.";
        } else if (tag === "Decentralized") {
            descriptionElement.textContent = "Decentralization distributes the decision-making from a central entity to multiple agents."
        } else if (tag === "Brainstorm") {
            descriptionElement.textContent = "Let's play some mind games!";
        }

        parentElement.insertBefore(descriptionElement, filteredList);
    }

    lists.forEach(listId => {
        const list = document.getElementById(listId);
        const items = Array.from(list.getElementsByTagName("li"));

        items.forEach(item => {
            const tags = item.dataset.tag ? item.dataset.tag.split(',') : [];
            if (tags.includes(tag)) {
                const clone = item.cloneNode(true);
                filteredList.appendChild(clone);
            }
        });
    });
}



function addTagAndDateInfo() {
    const lists = document.querySelectorAll("#rl-post-list li, #marl-post-list li");

    lists.forEach(item => {
        const annotateElement = item.querySelector("annotate");
        const date = item.getAttribute("data-date");
        const depth = item.getAttribute("data-depth");
        const tags = item.getAttribute("data-tag") || "NULL";
        const infoElement = document.createElement("span");
        infoElement.style.fontSize = "0.8em"; 
        infoElement.style.color = "#888888";
        infoElement.style.fontStyle = "italic";
        infoElement.innerHTML = `Tags: ${tags}; Specialty: ${depth}; Date: ${date}.`;
        annotateElement.insertAdjacentHTML("afterend", "<br>");
        annotateElement.insertAdjacentElement("afterend", infoElement);
        annotateElement.insertAdjacentHTML("afterend", "</br>");
    });
}
