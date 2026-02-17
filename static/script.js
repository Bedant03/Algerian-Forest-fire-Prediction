const content = `
<p>
Wildfires cause severe environmental and economic damage. 
Predicting fire intensity helps prevent large-scale disasters.
</p>

<p>
This system uses Machine Learning trained on the 
Algerian Forest Fire Dataset to predict the Fire Weather Index (FWI).
</p>

<p>
FWI estimates fire intensity based on weather conditions 
such as temperature, humidity, wind speed, and fuel moisture.
</p>

<ul style="text-align:left; margin-top:10px;">
<li>Identify high-risk areas</li>
<li>Improve disaster preparedness</li>
<li>Enable early warning systems</li>
<li>Support forest management decisions</li>
</ul>
`;

document.getElementById("dynamicContent").innerHTML = content;

function goToPrediction() {
    window.location.href = "/predict";
}