"use client"

import { useState } from "react";
import { BusinessCard } from "./BusinessCard";
import { Button } from "@mui/material";


interface BusinessExploreProps {
    businesses: Business[];
}

export const BusinessExplore = (props: BusinessExploreProps) => {
    const [currentB, setCurrentB] = useState<number>(0);

    const setBusinessRight = () => {
        if (currentB === props.businesses.length - 1) {
            setCurrentB(0);
        } else {
            setCurrentB(currentB + 1);
        }
    }

    const setBusinessLeft = () => {
        if (currentB === 0) {
            setCurrentB(props.businesses.length - 1);
        } else {
            setCurrentB(currentB - 1);
        }
    }

    return (
        <div>
            <BusinessCard business={props.businesses[currentB]} />
            <div className="button-container">
            <Button onClick={setBusinessLeft}>Left Arrow</Button>
            <Button onClick={setBusinessRight}>Right Arrow</Button>
            </div>
        </div>

    );
}