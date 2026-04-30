// ==============================
// CONFIGURACIÓN DE RANGOS CLÍNICOS
// ==============================
const RANGOS = {
  pregnancies:   { min: 0,    max: 17,   label: "Embarazos",        unidad: ""      },
  glucose:       { min: 44,   max: 199,  label: "Glucosa",          unidad: "mg/dL" },
  blood_pressure:{ min: 24,   max: 122,  label: "Presión arterial", unidad: "mmHg"  },
  skin_thickness:{ min: 7,    max: 99,   label: "Grosor de piel",   unidad: "mm"    },
  insulin:       { min: 14,   max: 846,  label: "Insulina",         unidad: "μU/mL" },
  bmi:           { min: 18,   max: 67,   label: "IMC",              unidad: "kg/m²" },
  dpf:           { min: 0.07, max: 2.42, label: "DPF",              unidad: ""      },
  age:           { min: 21,   max: 81,   label: "Edad",             unidad: "años"  },
};

// Mapeo: clave en el objeto data → id del campo en el DOM
const MAPA_IDS = {
  pregnancies:                "pregnancies",
  glucose:                    "glucose",
  blood_pressure:             "blood_pressure",
  skin_thickness:             "skin_thickness",
  insulin:                    "insulin",
  bmi:                        "bmi",
  diabetes_pedigree_function: "dpf",
  age:                        "age",
};

// ==============================
// HISTORIAL (en memoria — se reinicia al cerrar la pestaña)
// ==============================
let historial = [];
const MAX_HISTORIAL = 20;

function guardarEnHistorial(data, resultado) {
  const entrada = {
    id:          Date.now(),
    timestamp:   new Date(),
    inputs:      { ...data },
    porcentaje:  (resultado.probability * 100).toFixed(1),
    nivel:       resultado.probability >= 0.5 ? "alto"
                 : resultado.probability >= 0.2 ? "moderado"
                 : "bajo",
    riesgo:      resultado.risk || "—",
  };

  historial.unshift(entrada);          // más reciente primero
  if (historial.length > MAX_HISTORIAL) historial.pop();

  renderHistorial();
}

function renderHistorial() {
  const contenedor = document.getElementById("historial-contenedor");
  const lista      = document.getElementById("historial-lista");

  if (historial.length === 0) {
    contenedor.style.display = "none";
    return;
  }

  contenedor.style.display = "block";
  lista.innerHTML = "";

  historial.forEach((entrada, index) => {
    const hora = entrada.timestamp.toLocaleTimeString("es-MX", {
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
    const fecha = entrada.timestamp.toLocaleDateString("es-MX", {
      day: "2-digit", month: "short"
    });

    const iconos = { bajo: "✅", moderado: "🔶", alto: "⚠️" };
    const icono  = iconos[entrada.nivel] || "—";

    const fila = document.createElement("div");
    fila.className = `historial-fila historial-${entrada.nivel}`;
    fila.style.animationDelay = `${index * 0.04}s`;

    fila.innerHTML = `
      <div class="historial-fila-top">
        <div class="historial-resultado-grupo">
          <span class="historial-icono">${icono}</span>
          <span class="historial-porcentaje">${entrada.porcentaje}%</span>
          <span class="historial-badge badge-${entrada.nivel}">${entrada.nivel.charAt(0).toUpperCase() + entrada.nivel.slice(1)}</span>
        </div>
        <div class="historial-meta">
          <span class="historial-fecha">${fecha}</span>
          <span class="historial-hora">${hora}</span>
          <button class="btn-cargar" onclick="cargarDesdeHistorial(${entrada.id})" title="Cargar estos valores en el formulario">↩ Cargar</button>
        </div>
      </div>
      <div class="historial-valores">
        <span>G: <strong>${entrada.inputs.glucose}</strong></span>
        <span>PA: <strong>${entrada.inputs.blood_pressure}</strong></span>
        <span>IMC: <strong>${entrada.inputs.bmi}</strong></span>
        <span>Ins: <strong>${entrada.inputs.insulin}</strong></span>
        <span>Edad: <strong>${entrada.inputs.age}</strong></span>
        <span>DPF: <strong>${entrada.inputs.diabetes_pedigree_function}</strong></span>
      </div>
    `;

    lista.appendChild(fila);
  });
}

function cargarDesdeHistorial(id) {
  const entrada = historial.find(e => e.id === id);
  if (!entrada) return;

  // Mapeo inverso: clave del objeto data → id del input en el DOM
  const camposDom = {
    pregnancies:                "pregnancies",
    glucose:                    "glucose",
    blood_pressure:             "blood_pressure",
    skin_thickness:             "skin_thickness",
    insulin:                    "insulin",
    bmi:                        "bmi",
    diabetes_pedigree_function: "dpf",
    age:                        "age",
  };

  for (const [clave, domId] of Object.entries(camposDom)) {
    const input = document.getElementById(domId);
    if (input) {
      input.value = entrada.inputs[clave];
      limpiarError(domId);
    }
  }

  // Scroll al formulario
  document.querySelector(".form-grid").scrollIntoView({ behavior: "smooth", block: "start" });
}

function limpiarHistorial() {
  historial = [];
  renderHistorial();
}

// ==============================
// VALIDACIÓN EN VIVO
// ==============================
function validarCampoEnVivo(campoId) {
  const input = document.getElementById(campoId);
  const valor = parseFloat(input.value);
  const rango = RANGOS[campoId];

  if (input.value === "") {
    limpiarError(campoId);
    return true;
  }

  if (isNaN(valor)) {
    mostrarErrorCampo(campoId, "Ingresa un número válido.");
    return false;
  }

  if (valor < rango.min || valor > rango.max) {
    const unidad = rango.unidad ? ` ${rango.unidad}` : "";
    mostrarErrorCampo(campoId, `Valor fuera de rango clínico (${rango.min}–${rango.max}${unidad}).`);
    return false;
  }

  limpiarError(campoId);
  return true;
}

// ==============================
// VALIDACIÓN COMPLETA AL ENVIAR
// ==============================
function validarTodos(data) {
  let valido = true;
  let primerCampoConError = null;

  for (const [claveDato, campoId] of Object.entries(MAPA_IDS)) {
    const valor = data[claveDato];
    const rango = RANGOS[campoId];
    const input = document.getElementById(campoId);

    if (input.value === "" || input.value === null) {
      mostrarErrorCampo(campoId, "Este campo es obligatorio.");
      if (!primerCampoConError) primerCampoConError = input;
      valido = false;
      continue;
    }

    if (isNaN(valor)) {
      mostrarErrorCampo(campoId, "Ingresa un número válido.");
      if (!primerCampoConError) primerCampoConError = input;
      valido = false;
      continue;
    }

    if (valor < rango.min || valor > rango.max) {
      const unidad = rango.unidad ? ` ${rango.unidad}` : "";
      mostrarErrorCampo(campoId, `Valor fuera de rango clínico (${rango.min}–${rango.max}${unidad}).`);
      if (!primerCampoConError) primerCampoConError = input;
      valido = false;
    } else {
      limpiarError(campoId);
    }
  }

  if (primerCampoConError) {
    primerCampoConError.scrollIntoView({ behavior: "smooth", block: "center" });
    primerCampoConError.focus();
  }

  return valido;
}

function mostrarErrorCampo(campoId, mensaje) {
  const errEl = document.getElementById(`err-${campoId}`);
  const input = document.getElementById(campoId);
  if (errEl) errEl.textContent = mensaje;
  if (input) input.classList.add("input-error");
}

function limpiarError(campoId) {
  const errEl = document.getElementById(`err-${campoId}`);
  const input = document.getElementById(campoId);
  if (errEl) errEl.textContent = "";
  if (input) input.classList.remove("input-error");
}

// ==============================
// FUNCIÓN PRINCIPAL: CALCULAR RIESGO
// ==============================
async function calcularRiesgo() {
  const data = {
    pregnancies:                parseFloat(document.getElementById("pregnancies").value),
    glucose:                    parseFloat(document.getElementById("glucose").value),
    blood_pressure:             parseFloat(document.getElementById("blood_pressure").value),
    skin_thickness:             parseFloat(document.getElementById("skin_thickness").value),
    insulin:                    parseFloat(document.getElementById("insulin").value),
    bmi:                        parseFloat(document.getElementById("bmi").value),
    diabetes_pedigree_function: parseFloat(document.getElementById("dpf").value),
    age:                        parseFloat(document.getElementById("age").value),
  };

  if (!validarTodos(data)) return;

  const btn = document.getElementById("btn-calcular");
  btn.querySelector(".btn-text").style.display = "none";
  btn.querySelector(".btn-spinner").style.display = "inline";
  btn.disabled = true;

  document.getElementById("resultado").style.display = "none";
  document.getElementById("explicacion-contenedor").style.display = "none";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (result.error) {
      mostrarError("Error del servidor: " + result.error);
      return;
    }

    const porcentaje = result.probability ? (result.probability * 100).toFixed(1) : 0;
    const riesgo     = result.risk || "No disponible";

    let nivel = "bajo";
    let icono = "✅";
    if (result.probability >= 0.5) { nivel = "alto";     icono = "⚠️"; }
    else if (result.probability >= 0.2) { nivel = "moderado"; icono = "🔶"; }

    // Resultado principal
    const resultadoDiv = document.getElementById("resultado");
    resultadoDiv.style.display = "block";
    resultadoDiv.className = `resultado riesgo-${nivel}`;

    document.getElementById("resultado-icono").textContent = icono;
    document.getElementById("resultado-porcentaje").textContent = `${porcentaje}% de probabilidad`;
    document.getElementById("resultado-riesgo").textContent = riesgo;

    const barra = document.getElementById("barra-progreso");
    barra.style.width = "0%";
    barra.className = `barra-progreso barra-${nivel}`;
    setTimeout(() => { barra.style.width = `${porcentaje}%`; }, 50);

    // Análisis por factor
    if (result.explicacion && result.explicacion.length > 0) {
      const lista = document.getElementById("explicacion-lista");
      lista.innerHTML = "";

      result.explicacion.forEach((factor, index) => {
        const tarjeta = document.createElement("div");
        tarjeta.className = `factor-card factor-${factor.estado}`;
        tarjeta.style.animationDelay = `${index * 0.08}s`;

        const estadoLabel = {
          alto:     "⚠️ Alto",
          moderado: "🔶 Moderado",
          normal:   "✅ Normal",
        }[factor.estado] || factor.estado;

        tarjeta.innerHTML = `
          <div class="factor-header">
            <span class="factor-nombre">${factor.variable}</span>
            <span class="factor-badge badge-${factor.estado}">${estadoLabel}</span>
          </div>
          <div class="factor-mensaje">${factor.mensaje}</div>
        `;
        lista.appendChild(tarjeta);
      });

      document.getElementById("explicacion-contenedor").style.display = "block";
    }

    // ── GUARDAR EN HISTORIAL ──
    guardarEnHistorial(data, result);

    resultadoDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (error) {
    mostrarError("Error al conectar con el servidor. Verifica que la API esté corriendo.");
  } finally {
    btn.querySelector(".btn-text").style.display = "inline";
    btn.querySelector(".btn-spinner").style.display = "none";
    btn.disabled = false;
  }
}

function mostrarError(msg) {
  const resultadoDiv = document.getElementById("resultado");
  resultadoDiv.style.display = "block";
  resultadoDiv.className = "resultado riesgo-error";
  resultadoDiv.innerHTML = `<div style="text-align:center; padding: 10px;">❌ ${msg}</div>`;
}
