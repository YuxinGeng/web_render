/**
 * Evolutionary Dynamics with Vector Payoffs - Main JavaScript
 * 
 * This script handles:
 * 1. Form interactions and dynamic content generation
 * 2. Data submission to backend endpoints
 * 3. Visualization of results with D3.js
 */

// ================ UTILITY FUNCTIONS ================

/**
 * Converts binary representation to Cooperation/Defection notation
 * @param {number} binary - The binary number to convert
 * @param {number} K - Number of contexts/games
 * @returns {string} String of 'C's and 'D's representing the strategy
 */
function binary_to_CD(binary, K) {
    let binaryStr = binary.toString(2);
    if (binaryStr.length < K) {
        binaryStr = '0'.repeat(K - binaryStr.length) + binaryStr;
    }
    return binaryStr.split('').map(bit => bit === '0' ? 'C' : 'D').join('');
}

// ================ EVENT LISTENERS ================

// Listen for changes in the number of game contexts (K)
document.getElementById('K').addEventListener('input', function(event) {
    const K = parseInt(event.target.value, 10);
    const payoffContainer = document.getElementById('payoff-matrices');
    const NField = document.getElementById('N');
    
    payoffContainer.innerHTML = ''; // Clear previous content

    // Set default population size based on K
    if (K === 1) {
        NField.value = 100;
    } else if (K === 2) {
        NField.value = 8;
    } else if (K === 3) {
        NField.value = 6;
    }

    // Generate payoff matrix inputs for each context
    for (let i = 1; i <= K; i++) {
        const div = document.createElement('div');
        div.innerHTML = `
            <fieldset>
                <legend>Payoff matrix in context ${i}:</legend>
                \\(R_{${i}}\\) : <input type="text" name="R_${i}" step="any" required>
                \\(S_{${i}}\\) : <input type="text" name="S_${i}" step="any" required>
                \\(T_{${i}}\\) : <input type="text" name="T_${i}" step="any" required>
                \\(P_{${i}}\\) : <input type="text" name="P_${i}" step="any" required>
            </fieldset>
        `;
        payoffContainer.appendChild(div);
    }

    // Re-render MathJax formulas
    MathJax.typeset();
});

// Listen for changes in number of contexts for ODE simulation (Z)
document.getElementById('Z').addEventListener('input', function(event) {
    const Z = parseInt(event.target.value, 10);
    const lambdaContainer = document.getElementById('Lambda-inputs');
    const payoffContainer = document.getElementById('payoff-matrices_infty');
    const initContainer = document.getElementById('x0-inputs');
    
    // Clear previous content
    lambdaContainer.innerHTML = ''; 
    payoffContainer.innerHTML = ''; 
    initContainer.innerHTML = '';

    // Create lambda inputs
    const lambdaDiv = document.createElement('fieldset');
    const lambdaLegend = document.createElement('legend');
    lambdaLegend.innerHTML = `\\(\\lambda\\) vector:`;
    lambdaDiv.appendChild(lambdaLegend);

    const lambdaInputsContainer = document.createElement('div');

    // Create input for each strategy
    for (let i = 0; i < Math.pow(2, Z); i++) {
        const binaryLambda = binary_to_CD(i, Z);
        const inputLambda = document.createElement('input');
        inputLambda.type = 'text';
        inputLambda.name = `lambda_${i}`;
        inputLambda.value = '1';  // Default value
        inputLambda.required = true;
        inputLambda.style.marginRight = '10px';
        inputLambda.step = 'any';

        const lambdaRow = document.createElement('div');
        lambdaRow.style.display = 'inline-block';
        lambdaRow.style.marginBottom = '5px';
        lambdaRow.style.marginRight = '20px';
        lambdaRow.innerHTML = `\\(\\lambda_{\\text{${binaryLambda}}}\\) : `;
        lambdaRow.appendChild(inputLambda);
        lambdaInputsContainer.appendChild(lambdaRow);

        // Line break for better formatting when Z=3
        if (Z === 3 && (i === 3 || i === 7)) {
            lambdaInputsContainer.appendChild(document.createElement('br'));
        }
    }

    lambdaDiv.appendChild(lambdaInputsContainer);
    lambdaContainer.appendChild(lambdaDiv);

    // Create initial state inputs
    const x0Div = document.createElement('fieldset');
    const x0Legend = document.createElement('legend');
    x0Legend.innerHTML = `Initial state vector:`;
    x0Div.appendChild(x0Legend);

    const x0InputsContainer = document.createElement('div');

    // Create input for each strategy's initial state
    for (let i = 0; i < Math.pow(2, Z); i++) {
        const binaryX0 = binary_to_CD(i, Z);
        const inputX0 = document.createElement('input');
        inputX0.type = 'text';
        inputX0.name = `x0_${i}`;
        inputX0.value = (1 / Math.pow(2, Z)).toFixed(Z);  // Default value - equal distribution
        inputX0.required = true;
        inputX0.style.marginRight = '10px';
        inputX0.step = 'any';

        const x0Row = document.createElement('div');
        x0Row.style.display = 'inline-block';
        x0Row.style.marginBottom = '5px';
        x0Row.style.marginRight = '20px';
        x0Row.innerHTML = `\\(x_{\\text{${binaryX0}}}\\) : `;
        x0Row.appendChild(inputX0);
        x0InputsContainer.appendChild(x0Row);

        // Line break for better formatting when Z=3
        if (Z === 3 && (i === 3 || i === 7)) {
            x0InputsContainer.appendChild(document.createElement('br'));
        }
    }

    x0Div.appendChild(x0InputsContainer);
    initContainer.appendChild(x0Div);

    // Create payoff matrix inputs for each context
    for (let i = 1; i <= Z; i++) {
        const div = document.createElement('fieldset');
        const legend = document.createElement('legend');
        legend.innerHTML = `Payoff matrix in context ${i}:`;
        div.appendChild(legend);

        div.innerHTML += `
            \\(R_{${i}}\\) : <input type="text" name="RR_${i}" step="any" required>
            \\(S_{${i}}\\) : <input type="text" name="SS_${i}" step="any" required>
            \\(T_{${i}}\\) : <input type="text" name="TT_${i}" step="any" required>
            \\(P_{${i}}\\) : <input type="text" name="PP_${i}" step="any" required>
        `;
        payoffContainer.appendChild(div);
    }

    // Re-render MathJax formulas
    MathJax.typeset();
});

// ================ FORM SUBMISSIONS ================

// Handle ODE simulation form submission
document.getElementById('param-form-2').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const params = {};

    formData.forEach((value, key) => {
        params[key] = value;
    });

    fetch('/vectorpayoff/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        if (data.image) {
            // Display result in a new window
            let popupWindow = window.open("", "_blank", "width=1000,height=750");
            popupWindow.document.write(`
                <html>
                <head><title>Deterministic Dynamics in Infinite Populations</title></head>
                <body style="display: flex; justify-content: center; align-items: center; height: 100%; margin: 0;">
                    <img src="data:image/png;base64,${data.image}" alt="Deterministic Dynamics in Infinite Populations" style="max-width:100%; max-height:100%;">
                </body>
                </html>
            `);
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

// Handle Markov chain simulation form submission
document.getElementById('param-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const params = {};

    formData.forEach((value, key) => {
        params[key] = value;
    });

    fetch('/vectorpayoff/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        // Open a new window for the visualization
        const newWindow = window.open("", "_blank", "width=1000,height=750");
        if (newWindow) {
            // Set up the new window
            newWindow.document.title = "Markov Chain Visualization";
            newWindow.document.body.style.margin = "0";
            newWindow.document.body.style.overflow = "hidden";

            // Create the SVG element
            const svgElement = newWindow.document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svgElement.setAttribute("width", "100%");
            svgElement.setAttribute("height", "100%");
            svgElement.setAttribute("viewBox", "0 0 4000 3000");
            newWindow.document.body.appendChild(svgElement);

            // Initialize the visualization
            const svg = d3.select(newWindow.document.querySelector("svg"));
            updateVisualization(svg, data.nodes, data.edges);
        }
    })
    .catch(error => console.error('Error:', error));
});

// ================ VISUALIZATION FUNCTIONS ================

/**
 * Creates a D3.js visualization of nodes and edges
 * @param {Object} svg - D3 selection of the SVG element
 * @param {Array} nodesData - Array of node objects
 * @param {Array} edgesData - Array of edge objects
 */
function updateVisualization(svg, nodesData, edgesData) {
    const g = svg.append("g");

    // Set up D3 force simulation
    const simulation = d3.forceSimulation(nodesData)
        .force("link", d3.forceLink(edgesData).id(d => d.id).distance(50))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(2000, 1000))
        .force("collide", d3.forceCollide().radius(d => d.size_infty + 10))
        .force("attractForce", d3.forceRadial(0, svg.attr("viewBox").split(" ")[2] / 2, svg.attr("viewBox").split(" ")[3] / 2).strength(0.1));

    // Create edges
    const link = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(edgesData)
        .enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", d => d.weight * 5)
        .attr("stroke", d => d.color);

    // Create nodes
    const node = g.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(nodesData)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", d => d.size_infty)
        .attr("fill", d => d.color)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Create labels
    const label = g.append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(nodesData)
        .enter().append("text")
        .attr("dy", 5)
        .attr("dx", -35)
        .text(d => d.label)
        .style("user-select", "none") // Prevent text selection
        .on("mousedown", function(event, d) {
            event.stopPropagation(); // Prevent zoom behavior when clicking labels
        })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Set up zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoom);

    // Update positions on each tick
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        label
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    });

    // Drag handlers for nodes
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}
