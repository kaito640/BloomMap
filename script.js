// Load cluster data and global summary for the word cloud
Promise.all([
  d3.csv("cluster_summary.csv"),
  d3.csv("global_summary.csv")
]).then(([clusterData, globalData]) => {
  
  // Prepare the SVG canvas
  const width = 960, height = 960;
  const radius = Math.min(width, height) / 2;
  
  const svg = d3.select("svg")
    .attr("width", width)
    .attr("height", height)
  .append("g")
    .attr("transform", `translate(${width / 2}, ${height / 2})`);
  
  // Group data by clusters
  const clusters = d3.group(clusterData, d => d.cluster);
  
  // Create a scale for the angles
  const angleScale = d3.scaleLinear()
    .domain([0, clusters.size])
    .range([0, 2 * Math.PI]);
  
  const clusterRadius = radius / 3;  // To position clusters closer to the center
  
  // Create an arc generator
  const arc = d3.arc()
    .innerRadius(clusterRadius - 20)
    .outerRadius(clusterRadius + 20)
    .startAngle((d, i) => angleScale(i))
    .endAngle((d, i) => angleScale(i + 1));
  
  // Draw each cluster as an arc
  svg.selectAll(".cluster")
    .data(Array.from(clusters.keys()))
    .enter().append("path")
      .attr("class", "cluster")
      .attr("d", (d, i) => arc(d, i))
      .attr("fill", (d, i) => d3.schemeCategory10[i % 10])
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
  
  // Add text for each cluster centered on the arc
  svg.selectAll(".cluster-text")
    .data(Array.from(clusters.keys()))
    .enter().append("text")
      .attr("transform", (d, i) => {
        const [x, y] = arc.centroid(d, i);  // Get centroid of the arc
        return `translate(${x}, ${y})`;
      })
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .text(d => d)
      .style("font-size", "14px");  // Adjust the font size as needed
  
  // ======== WORD CLOUD IN THE CENTER ========

  // Define the word cloud layout
  const wordCloudRadius = clusterRadius - 100; // Adjust size for the word cloud
  
  const topXWords = globalData.slice(0, 50); // X most frequent words, modify as needed
  
  const layout = d3.layout.cloud()
    .size([wordCloudRadius * 2, wordCloudRadius * 2])
    .words(topXWords.map(d => ({text: d.word, size: +d.frequency})))
    .padding(5)
    .fontSize(d => Math.log(d.size) * 10) // Scale font size based on frequency
    .on("end", drawWordCloud);
  
  layout.start();
  
  function drawWordCloud(words) {
    svg.append("g")
      .attr("transform", `translate(${0}, ${0})`)  // Center the word cloud
      .selectAll("text")
      .data(words)
      .enter().append("text")
        .style("font-size", d => `${d.size}px`)
        .style("fill", () => d3.schemeCategory10[Math.floor(Math.random() * 10)])
        .attr("text-anchor", "middle")
        .attr("transform", d => `translate(${[d.x, d.y]}) rotate(${d.rotate})`)
        .text(d => d.text);
  }

});

