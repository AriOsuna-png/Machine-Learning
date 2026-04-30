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
// VALIDACIÓN EN VIVO (mientras el usuario escribe)
// ==============================
function validarCampoEnVivo(campoId) {
  const input = document.getElementById(campoId);
  const valor = parseFloat(input.value);
  const rango = RANGOS[campoId];

  // Campo vacío: sin error aún (no molestar mientras escribe)
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
    mostrarErrorCampo(
      campoId,
      `Valor fuera de rango clínico (${rango.min}–${rango.max}${unidad}).`
    );
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

    // Campo vacío
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
      mostrarErrorCampo(
        campoId,
        `Valor fuera de rango clínico (${rango.min}–${rango.max}${unidad}).`
      );
      if (!primerCampoConError) primerCampoConError = input;
      valido = false;
    } else {
      limpiarError(campoId);
    }
  }

  // Scroll al primer campo con error
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

  // ── VALIDACIÓN ANTES DE ENVIAR ──
  if (!validarTodos(data)) return;

  // Mostrar estado de carga
  const btn = document.getElementById("btn-calcular");
  btn.querySelector(".btn-text").style.display = "none";
  btn.querySelector(".btn-spinner").style.display = "inline";
  btn.disabled = true;

  // Ocultar resultados anteriores
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
    const riesgo = result.risk || "No disponible";

    // Determinar nivel de riesgo
    let nivel = "bajo";
    let icono = "✅";
    if (result.probability >= 0.5) {
      nivel = "alto";
      icono = "⚠️";
    } else if (result.probability >= 0.2) {
      nivel = "moderado";
      icono = "🔶";
    }

    // Mostrar resultado principal
    const resultadoDiv = document.getElementById("resultado");
    resultadoDiv.style.display = "block";
    resultadoDiv.className = `resultado riesgo-${nivel}`;

    document.getElementById("resultado-icono").textContent = icono;
    document.getElementById("resultado-porcentaje").textContent = `${porcentaje}% de probabilidad`;
    document.getElementById("resultado-riesgo").textContent = riesgo;

    // Barra de progreso animada
    const barra = document.getElementById("barra-progreso");
    barra.style.width = "0%";
    barra.className = `barra-progreso barra-${nivel}`;
    setTimeout(() => {
      barra.style.width = `${porcentaje}%`;
    }, 50);

    // Mostrar explicación por factor
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

    // Scroll suave al resultado
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
